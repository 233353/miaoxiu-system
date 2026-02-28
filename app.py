from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import webbrowser
import threading
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
from functools import wraps

import shutil
import zipfile
from werkzeug.security import generate_password_hash, check_password_hash

import paddle
from paddle.vision.models import resnet50
import paddle.nn as nn
import numpy as np
from sklearn.cluster import KMeans

import os
from dotenv import load_dotenv


# 创建flask应用实例
app = Flask(__name__)

load_dotenv()

# 应用配置
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set")

app.config.update(
    SECRET_KEY=SECRET_KEY,
    UPLOAD_FOLDER='static/uploads',
    DATABASE='database.db',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=3600,
)


def get_users_from_db():
    """从数据库获取用户（备份函数）"""
    try:
        conn = get_db_connection()
        users = conn.execute('SELECT username, password_hash FROM users').fetchall()
        conn.close()

        user_dict = {}
        for user in users:
            user_dict[user['username']] = user['password_hash']
        return user_dict
    except:
        # 如果数据库出错，返回空字典
        return {}


# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# 登录保护装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('请先登录以访问此页面！', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# 登录/登出路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    # 如果已经登录，直接跳转到首页
    if session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # 从数据库验证用户
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()

        if user and verify_password(password, user['password_hash']):
            session['user'] = user['username']
            session['user_id'] = user['id']
            session['user_role'] = user['role']
            session['logged_in'] = True
            session['real_name'] = user['real_name'] or user['username']

            flash(f'欢迎回来，{session["real_name"]}！', 'success')

            # 如果有next参数，跳转到目标页面
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('用户名或密码错误！', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('logged_in', None)
    flash('您已成功退出登录', 'success')
    return redirect(url_for('index'))


# 图片预处理
def preprocess_image(image_path):
    """手动实现图片预处理"""
    try:
        # 打开图片
        image = Image.open(image_path).convert('RGB')

        # 调整大小和中心裁剪
        image = image.resize((256, 256))
        width, height = image.size
        left = (width - 224) // 2
        top = (height - 224) // 2
        right = left + 224
        bottom = top + 224
        image = image.crop((left, top, right, bottom))

        # 转换为numpy数组并归一化
        image_array = np.array(image).astype(np.float32) / 255.0

        # 标准化 (使用ImageNet的均值和标准差)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        # 调整维度顺序为 CHW
        image_array = image_array.transpose(2, 0, 1)

        return image_array
    except Exception as e:
        print(f"图片预处理错误: {e}")
        return None


# 初始化模型
try:
    model = resnet50(pretrained=True)
    model.eval()
    # 创建特征提取模型（移除最后的分类层）
    feature_model = nn.Sequential(*list(model.children())[:-1])
    print("PaddlePaddle模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    feature_model = None


# 数据库函数
def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    # 创建patterns表
    conn.execute('''
    CREATE TABLE IF NOT EXISTS patterns(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        region TEXT,
        inheritor TEXT,
        tags TEXT,
        description TEXT,
        image_path TEXT NOT NULL,
        feature_vector BLOB, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 创建users表
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        real_name TEXT,
        phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 创建backup_records表
    conn.execute('''
    CREATE TABLE IF NOT EXISTS backup_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        backup_path VARCHAR(500) NOT NULL,
        backup_type VARCHAR(50) NOT NULL,
        backup_size BIGINT,
        status VARCHAR(20) NOT NULL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')

    try:
        # 如果现有数据是字符串格式，转换为BLOB
        patterns = conn.execute('SELECT id, feature_vector FROM patterns WHERE feature_vector IS NOT NULL').fetchall()
        for pattern in patterns:
            if isinstance(pattern['feature_vector'], str):
                # 将字符串转换为字节
                feature_bytes = pickle.dumps([float(x) for x in pattern['feature_vector'].split(',')])
                conn.execute('UPDATE patterns SET feature_vector = ? WHERE id = ?',
                             (feature_bytes, pattern['id']))
    except Exception as e:
        print(f"数据库迁移错误: {e}")

    try:
        # 创建空的admin账户（没有密码，需要用户自己设置）
        conn.execute(
            'INSERT OR IGNORE INTO users (username, password_hash, role, real_name) VALUES (?, ?, ?, ?)',
            ('admin', '', 'admin', '系统管理员')
        )
        conn.commit()

        print("=" * 50)
        print("重要提示")
        print("=" * 50)
        print("管理员账户 'admin' 已创建")
        print("但密码为空，无法登录！")
        print("")
        print("请运行以下脚本之一设置密码：")
        print("1. reset_admin_github.py  (GitHub用户)")
        print("2. reset_admin.py")
        print("=" * 50)
    except Exception as e:
        print(f"初始化用户数据错误: {e}")

    conn.commit()
    conn.close()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


# 密码相关函数
def verify_password(password, password_hash):
    """验证密码"""
    return check_password_hash(password_hash, password)


def hash_password(password):
    """密码哈希"""
    return generate_password_hash(password)


# 备份相关函数
def find_backup_devices():
    """查找可用的备份设备（U盘等）"""
    devices = []

    # Windows 系统检测
    if os.name == 'nt':
        import string
        drives = []
        for drive in string.ascii_uppercase:
            drive_path = f"{drive}:\\"
            if os.path.exists(drive_path):
                try:
                    # 获取驱动器信息
                    total, used, free = shutil.disk_usage(drive_path)
                    drive_info = {
                        'path': drive_path,
                        'name': f"本地磁盘 {drive}",
                        'free_space': f"{free // (2 ** 30)} GB",
                        'total_space': f"{total // (2 ** 30)} GB"
                    }
                    drives.append(drive_info)
                except:
                    continue

        # 优先显示可移动设备（通常是U盘）
        removable_drives = [d for d in drives if d['path'] in ['D:\\', 'E:\\', 'F:\\', 'G:\\']]
        fixed_drives = [d for d in drives if d['path'] in ['C:\\']]

        devices = removable_drives + fixed_drives

    # 如果没有找到设备，提供默认选项
    if not devices:
        devices = [{
            'path': './backups',
            'name': '本地备份文件夹',
            'free_space': '未知',
            'total_space': '未知'
        }]

    return devices


def perform_backup(backup_path):
    """执行备份操作"""
    try:
        # 创建备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"miaoxiu_backup_{timestamp}.zip"
        backup_filepath = os.path.join(backup_path, backup_filename)

        # 确保备份目录存在
        os.makedirs(backup_path, exist_ok=True)

        # 创建ZIP备份文件
        with zipfile.ZipFile(backup_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 备份数据库
            if os.path.exists(app.config['DATABASE']):
                zipf.write(app.config['DATABASE'], 'database.db')

            # 备份上传的图片
            uploads_dir = app.config['UPLOAD_FOLDER']
            if os.path.exists(uploads_dir):
                for root, dirs, files in os.walk(uploads_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(uploads_dir))
                        zipf.write(file_path, arcname)

                # 获取备份文件大小（字节）
                backup_size = os.path.getsize(backup_filepath)

                # === 保存备份记录到数据库 ===
                conn = get_db_connection()
                conn.execute('''
                INSERT INTO backup_records 
                (user_id, backup_path, backup_type, backup_size, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    session.get('user_id', 0),
                    backup_filepath,
                    'manual',
                    backup_size,
                    'success',
                    '用户手动创建的备份文件'
                ))
                conn.commit()
                conn.close()

                return {
                    'success': True,
                    'backup_file': backup_filename,
                    'file_size': backup_size,
                    'backup_path': backup_filepath
                }

    except Exception as e:
        conn = get_db_connection()
        conn.execute('''
        INSERT INTO backup_records 
        (user_id, backup_path, backup_type, status, notes)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            session.get('user_id', 0),
            backup_path,
            'manual',
            'failed',
            f'备份失败: {str(e)}'
        ))
        conn.commit()
        conn.close()

        return {'success': False, 'error': f'备份失败: {str(e)}'}


def perform_restore(backup_file):
    """执行恢复操作"""
    try:
        # 保存上传的备份文件
        temp_backup_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_backup.zip')
        backup_file.save(temp_backup_path)

        # 验证ZIP文件
        if not zipfile.is_zipfile(temp_backup_path):
            return {'success': False, 'error': '无效的备份文件'}

        # 获取备份文件信息
        backup_size = os.path.getsize(temp_backup_path)
        original_filename = backup_file.filename

        # 创建恢复目录
        restore_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'restore_temp')
        os.makedirs(restore_dir, exist_ok=True)

        # 解压备份文件
        with zipfile.ZipFile(temp_backup_path, 'r') as zipf:
            zipf.extractall(restore_dir)

        # 备份当前数据库
        if os.path.exists(app.config['DATABASE']):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_current_db = f"database_backup_before_restore_{timestamp}.db"
            shutil.copy2(app.config['DATABASE'],
                         os.path.join(app.config['UPLOAD_FOLDER'], backup_current_db))

        # 恢复数据库
        db_backup_path = os.path.join(restore_dir, 'database.db')
        if os.path.exists(db_backup_path):
            shutil.copy2(db_backup_path, app.config['DATABASE'])

        # 恢复图片文件
        for item in os.listdir(restore_dir):
            if item != 'database.db':
                src = os.path.join(restore_dir, item)
                dst = os.path.join(app.config['UPLOAD_FOLDER'], item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        # 记录恢复操作到备份记录表
        conn = get_db_connection()
        conn.execute('''
        INSERT INTO backup_records 
        (user_id, backup_path, backup_type, backup_size, status, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session.get('user_id', 0),
            original_filename,
            'restore',
            backup_size,
            'success',
            f'从备份文件恢复系统数据: {original_filename}'  # notes
        ))
        conn.commit()
        conn.close()

        # 清理临时文件
        if os.path.exists(temp_backup_path):
            os.remove(temp_backup_path)
        if os.path.exists(restore_dir):
            shutil.rmtree(restore_dir)

        return {'success': True}

    except Exception as e:
        # 记录恢复失败
        conn = get_db_connection()
        conn.execute('''
        INSERT INTO backup_records 
        (user_id, backup_path, backup_type, status, notes)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            session.get('user_id', 0),
            backup_file.filename if hasattr(backup_file, 'filename') else 'unknown',
            'restore',
            'failed',
            f'恢复失败: {str(e)}'
        ))
        conn.commit()
        conn.close()

        return {'success': False, 'error': f'恢复失败: {str(e)}'}


# 色彩提取函数
def extract_dominant_colors(image_path, num_colors=5):
    """提取图片主色调"""
    try:
        # 使用PIL打开图片
        image = Image.open(image_path)
        # 缩小图片加速处理
        image = image.resize((100, 100))
        # 转换为numpy数组
        pixels = np.array(image).reshape(-1, 3)

        # 使用K-means聚类找到主色调
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)

        # 获取聚类中心和比例
        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        percentages = (counts / len(pixels) * 100).round(2)

        # 按比例排序
        color_data = []
        for i in range(num_colors):
            color_data.append({
                'rgb': colors[i].tolist(),
                'hex': rgb_to_hex(colors[i]),
                'percentage': percentages[i]
            })

        # 按比例从高到低排序
        color_data.sort(key=lambda x: x['percentage'], reverse=True)

        return color_data

    except Exception as e:
        print(f"色彩提取错误: {e}")
        # 返回模拟数据
        return [
            {'rgb': [120, 80, 60], 'hex': '#78503C', 'percentage': 35.0},
            {'rgb': [200, 150, 100], 'hex': '#C89664', 'percentage': 25.0},
            {'rgb': [80, 120, 80], 'hex': '#507850', 'percentage': 20.0},
            {'rgb': [180, 180, 200], 'hex': '#B4B4C8', 'percentage': 15.0},
            {'rgb': [220, 200, 180], 'hex': '#DCC8B4', 'percentage': 5.0}
        ]


def rgb_to_hex(rgb):
    """RGB转十六进制"""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


# AI相关函数
def extract_features(image_path):
    """提取图片特征向量 """
    try:
        print(f"开始提取特征:{image_path}")

        # 如果模型加载失败，使用模拟特征
        if feature_model is None:
            return [float(np.random.randn()) for _ in range(2048)]

        # 最简单的预处理
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0

        # 转换为paddle tensor
        image_tensor = paddle.to_tensor(image_array.transpose(2, 0, 1), dtype='float32')
        image_tensor = paddle.unsqueeze(image_tensor, axis=0)

        # 提取特征
        with paddle.no_grad():
            features = feature_model(image_tensor)

        # 确保返回float32
        feature_vector = features.flatten().numpy().astype(np.float32).tolist()
        print(f"特征提取成功，向量长度:{len(feature_vector)}")
        return feature_vector

    except Exception as e:
        print(f"特征提取失败:{e}")
        # 返回模拟特征向量
        return [float(np.random.randn()) for _ in range(2048)]


def calculate_similarity(vec1, vec2):
    """计算两个特征向量的余弦相似度 - 改进版本"""
    try:
        # 转换为numpy数组
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)

        # 归一化向量（重要！）
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)  # 避免除零
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

        # 计算余弦相似度
        similarity = np.dot(vec1_norm, vec2_norm)

        # 确保在合理范围内
        similarity = np.clip(similarity, 0.0, 1.0)

        similarity_percent = round(similarity * 100, 2)

        print(f"相似度计算: 原始={similarity:.6f}, 百分比={similarity_percent}%")
        return similarity_percent

    except Exception as e:
        print(f"相似度计算错误：{e}")
        return 0.0


@app.route('/')
def index():
    conn = get_db_connection()
    patterns = conn.execute('SELECT * FROM patterns ORDER BY created_at DESC').fetchall()
    conn.close()

    patterns_list = []
    for pattern in patterns:
        pattern_dict = dict(pattern)
        pattern_dict['image_url'] = url_for('static', filename=f"uploads/{os.path.basename(pattern['image_path'])}")
        patterns_list.append(pattern_dict)

    return render_template('index.html', patterns=patterns_list)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('没有选择文件')
            return render_template('upload.html')

        file = request.files['image']
        if file.filename == '':
            flash('没有选择文件')
            return render_template('upload.html')

        if file and allowed_file(file.filename):
            try:
                # 保存图片
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # 提取向量特征
                print("开始提取图片特征...")
                feature_vector = extract_features(filepath)
                # 用pickle序列化特征向量为二进制
                feature_blob = pickle.dumps(feature_vector)
                print(f"特征提取完成，向量长度：{len(feature_vector)}")

                conn = get_db_connection()
                conn.execute(
                 '''INSERT INTO patterns(name, region, inheritor, tags, description, image_path, feature_vector) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (
                        request.form.get('name', '未命名纹样'),
                        request.form.get('region', '未知'),
                        request.form.get('inheritor', '未知'),
                        request.form.get('tags', ''),
                        request.form.get('description', ''),
                        filename,
                        feature_blob
                     )
                )
                conn.commit()
                conn.close()

                flash('纹样上传成功！')
                return redirect(url_for('index'))

            except Exception as e:
                flash(f'上传失败: {str(e)}')
                return render_template('upload.html')
        else:
            flash('不支持的文件格式')
            return render_template('upload.html')

    return render_template('upload.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        if 'search_image' not in request.files:
            return render_template('search.html',
                                   search_performed=True,
                                   error='没有选择搜索图片',
                                   similar_patterns=[])

        file = request.files['search_image']
        if file.filename == '':
            return render_template('search.html',
                                   search_performed=True,
                                   error='没有选择搜索图片',
                                   similar_patterns=[])

        if file and allowed_file(file.filename):
            try:
                # 保存临时搜索图片
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = f"search_{timestamp}_{filename}"
                temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                file.save(temp_filepath)

                # 提取搜索图片的特征
                print("提取搜索图片特征...")
                search_features = extract_features(temp_filepath)

                print("\n=== 开始相似度诊断 ===")

                # 自检：搜索图片与自身的相似度
                self_similarity = calculate_similarity(search_features, search_features)
                print(f"自检相似度（应该接近100%）: {self_similarity}%")

                # 如果自检相似度不正常，说明特征提取有问题
                if self_similarity < 99:
                    print(f"警告：自检相似度异常，可能特征提取有问题")

                # 详细诊断函数
                def debug_similarity(vec1, vec2, name1="vec1", name2="vec2"):
                    """详细诊断相似度计算"""
                    vec1 = np.array(vec1, dtype=np.float32)
                    vec2 = np.array(vec2, dtype=np.float32)

                    print(f"\n=== 相似度详细诊断 ===")
                    print(f"{name1} 长度: {len(vec1)}, 范围: [{vec1.min():.6f}, {vec1.max():.6f}]")
                    print(f"{name2} 长度: {len(vec2)}, 范围: [{vec2.min():.6f}, {vec2.max():.6f}]")

                    # 检查NaN和Inf
                    print(f"{name1} NaN: {np.isnan(vec1).any()}, Inf: {np.isinf(vec1).any()}")
                    print(f"{name2} NaN: {np.isnan(vec2).any()}, Inf: {np.isinf(vec2).any()}")

                    # 归一化
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    print(f"范数: {name1}={norm1:.6f}, {name2}={norm2:.6f}")

                    vec1_norm = vec1 / (norm1 + 1e-8)
                    vec2_norm = vec2 / (norm2 + 1e-8)

                    dot_product = np.dot(vec1_norm, vec2_norm)
                    print(f"点积: {dot_product:.6f}")

                    return dot_product

                # 对搜索图片自身进行详细诊断
                debug_similarity(search_features, search_features, "搜索特征", "搜索特征")
                print("=== 诊断结束 ===\n")

                # 获取数据库中的所有纹样
                conn = get_db_connection()
                all_patterns = conn.execute('SELECT * FROM patterns').fetchall()
                conn.close()

                # 获取筛选参数
                region_filter = request.form.get('region_filter', '')
                inheritor_filter = request.form.get('inheritor_filter', '')
                tags_filter = request.form.get('tags_filter', '')

                print(f"筛选条件 - 地区: {region_filter}, 传承人: {inheritor_filter}, 标签: {tags_filter}")

                # 计算相似度
                similar_patterns = []
                for pattern in all_patterns:
                    if pattern['feature_vector']:  # 确保有特征向量
                        # 从BLOB反序列化特征向量
                        db_features = pickle.loads(pattern['feature_vector'])
                        similarity = calculate_similarity(search_features, db_features)

                        # 只保留相似度较高的结果，并符合筛选条件
                        if similarity > 50:  # 50%相似度阈值
                            # 筛选条件
                            region_match = not region_filter or (
                                        pattern['region'] and region_filter in pattern['region'])
                            inheritor_match = not inheritor_filter or (
                                        pattern['inheritor'] and inheritor_filter.lower() in pattern[
                                    'inheritor'].lower())
                            tags_match = not tags_filter or (
                                        pattern['tags'] and tags_filter.lower() in pattern['tags'].lower())

                            if region_match and inheritor_match and tags_match:
                                pattern_dict = dict(pattern)
                                pattern_dict['image_url'] = f"uploads/{os.path.basename(pattern['image_path'])}"
                                pattern_dict['similarity'] = similarity
                                similar_patterns.append(pattern_dict)

                        # 按相似度排序（从高到低）
                    similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)

                # 删除临时文件
                os.remove(temp_filepath)

                return render_template('search.html',
                                       search_performed=True,
                                       similar_patterns=similar_patterns)

            except Exception as e:
                print(f"搜索过程中出错: {e}")
                return render_template('search.html',
                                       search_performed=True,
                                       error=f"搜索过程中出错: {e}",
                                       similar_patterns=[])
        else:
            return render_template('search.html',
                                   search_performed=True,
                                   error='不支持的文件格式',
                                   similar_patterns=[])

    # GET 请求 - 显示空搜索页面
    return render_template('search.html', search_performed=False)


@app.route('/pattern/<int:pattern_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_pattern(pattern_id):
    """编辑纹样"""
    conn = get_db_connection()
    pattern = conn.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,)).fetchone()

    if not pattern:
        conn.close()
        flash('纹样不存在')
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            # 获取表单数据
            name = request.form.get('name', '').strip()
            region = request.form.get('region', '').strip()
            inheritor = request.form.get('inheritor', '').strip()
            tags = request.form.get('tags', '').strip()
            description = request.form.get('description', '').strip()

            # 处理新上传的图片（如果有）
            new_image_filename = None
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename != '' and allowed_file(file.filename):
                    # 保存新图片
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_image_filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_image_filename)
                    file.save(filepath)

                    # 提取新图片的特征
                    feature_vector = extract_features(filepath)
                    feature_blob = pickle.dumps(feature_vector)

                    # 更新图片路径和特征
                    conn.execute('''
                    UPDATE patterns 
                    SET name = ?, region = ?, inheritor = ?, tags = ?, 
                        description = ?, image_path = ?, feature_vector = ?
                    WHERE id = ?
                    ''', (name, region, inheritor, tags, description,
                          new_image_filename, feature_blob, pattern_id))
                else:
                    # 只更新文本信息，不更新图片
                    conn.execute('''
                    UPDATE patterns 
                    SET name = ?, region = ?, inheritor = ?, tags = ?, 
                        description = ?
                    WHERE id = ?
                    ''', (name, region, inheritor, tags, description, pattern_id))
            else:
                # 只更新文本信息
                conn.execute('''
                UPDATE patterns 
                SET name = ?, region = ?, inheritor = ?, tags = ?, 
                    description = ?
                WHERE id = ?
                ''', (name, region, inheritor, tags, description, pattern_id))

            conn.commit()
            conn.close()
            flash('纹样信息更新成功！', 'success')
            return redirect(url_for('pattern_detail', pattern_id=pattern_id))

        except Exception as e:
            conn.close()
            flash(f'更新失败: {str(e)}', 'error')
            return render_template('edit_pattern.html', pattern=dict(pattern))

    # GET请求：显示编辑表单
    conn.close()
    pattern_dict = dict(pattern)
    pattern_dict['image_url'] = f"uploads/{os.path.basename(pattern['image_path'])}"
    return render_template('edit_pattern.html', pattern=pattern_dict)


@app.route('/pattern/<int:pattern_id>/delete', methods=['POST'])
@login_required
def delete_pattern(pattern_id):
    """删除纹样"""
    try:
        conn = get_db_connection()

        # 获取纹样信息（用于删除图片文件）
        pattern = conn.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,)).fetchone()

        if not pattern:
            conn.close()
            return {'success': False, 'error': '纹样不存在'}

        # 删除纹样记录
        conn.execute('DELETE FROM patterns WHERE id = ?', (pattern_id,))
        conn.commit()
        conn.close()

        # 删除对应的图片文件
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], pattern['image_path'])
        if os.path.exists(image_path):
            os.remove(image_path)

        return {'success': True, 'message': '纹样删除成功'}

    except Exception as e:
        return {'success': False, 'error': f'删除失败: {str(e)}'}


@app.route('/pattern/<int:pattern_id>')
def pattern_detail(pattern_id):
    """纹样详情页面"""
    conn = get_db_connection()
    pattern = conn.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,)).fetchone()
    conn.close()

    if pattern:
        pattern_dict = dict(pattern)
        pattern_dict['image_url'] = f"uploads/{os.path.basename(pattern['image_path'])}"
        return render_template('detail.html', pattern=pattern_dict)
    else:
        flash('纹样不存在')
        return redirect(url_for('index'))


@app.route('/analysis')
def analysis():
    """纹样分析页面"""
    return render_template('analysis.html')


@app.route('/analyze_colors', methods=['POST'])
def analyze_colors():
    """分析图片色彩"""
    if 'image' not in request.files:
        return {'error': '没有选择图片'}, 400

    file = request.files['image']
    if file.filename == '':
        return {'error': '没有选择图片'}, 400

    if file and allowed_file(file.filename):
        try:
            # 保存临时图片
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"color_analysis_{timestamp}_{filename}"
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            file.save(temp_filepath)

            # 提取色彩
            colors = extract_dominant_colors(temp_filepath)

            # 删除临时文件
            os.remove(temp_filepath)

            return {'success': True, 'colors': colors}

        except Exception as e:
            return {'error': f'色彩分析失败: {str(e)}'}, 500
    else:
        return {'error': '不支持的文件格式'}, 400


@app.route('/compare_patterns', methods=['POST'])
def compare_patterns():
    """比较两个纹样"""
    print("=== 开始纹样对比 ===")

    if 'image1' not in request.files or 'image2' not in request.files:
        print("缺少图片文件")
        return {'error': '请选择两张图片'}, 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        print("文件名为空")
        return {'error': '请选择两张图片'}, 400

    if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
        try:
            # 保存临时图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 第一张图片
            temp_file1 = f"compare1_{timestamp}_{secure_filename(file1.filename)}"
            temp_filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], temp_file1)
            file1.save(temp_filepath1)
            print(f"保存图片1: {temp_file1}")

            # 第二张图片
            temp_file2 = f"compare2_{timestamp}_{secure_filename(file2.filename)}"
            temp_filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], temp_file2)
            file2.save(temp_filepath2)
            print(f"保存图片2: {temp_file2}")

            # 提取特征并计算相似度
            print("开始提取特征...")
            features1 = extract_features(temp_filepath1)
            features2 = extract_features(temp_filepath2)

            print("开始计算相似度...")
            similarity = calculate_similarity(features1, features2)
            print(f"相似度计算结果: {similarity}%")

            # 生成对比结果描述
            if similarity >= 90:
                message = f"两个纹样非常相似！相似度: {similarity}%"
            elif similarity >= 70:
                message = f"两个纹样比较相似，相似度: {similarity}%"
            elif similarity >= 50:
                message = f"两个纹样有一定相似性，相似度: {similarity}%"
            else:
                message = f"两个纹样差异较大，相似度: {similarity}%"

            # 删除临时文件
            os.remove(temp_filepath1)
            os.remove(temp_filepath2)
            print("临时文件已清理")

            return {
                'success': True,
                'similarity': similarity,
                'message': message,
                'level': get_similarity_level(similarity)
            }

        except Exception as e:
            print(f"纹样对比失败: {e}")
            # 如果出错，尝试删除可能存在的临时文件
            try:
                if 'temp_filepath1' in locals() and os.path.exists(temp_filepath1):
                    os.remove(temp_filepath1)
                if 'temp_filepath2' in locals() and os.path.exists(temp_filepath2):
                    os.remove(temp_filepath2)
            except:
                pass

            return {'error': f'纹样对比失败: {str(e)}'}, 500
    else:
        print("不支持的文件格式")
        return {'error': '不支持的文件格式'}, 400


def get_similarity_level(similarity):
    """获取相似度等级"""
    if similarity >= 90:
        return "非常高"
    elif similarity >= 70:
        return "较高"
    elif similarity >= 50:
        return "中等"
    elif similarity >= 30:
        return "较低"
    else:
        return "很低"


# 备份功能路由
@app.route('/backup')
@login_required
def backup():
    """数据备份页面 - 现在包括备份和恢复"""
    # 检查权限
    if session.get('user_role') != 'admin':
        flash('只有管理员可以访问备份功能', 'error')
        return redirect(url_for('index'))

    # 获取可用的备份设备
    backup_devices = find_backup_devices()

    # 获取最近的备份记录（前5条）
    conn = get_db_connection()
    recent_records = conn.execute('''
    SELECT id, backup_path, backup_type, backup_size, status, created_at
    FROM backup_records 
    ORDER BY created_at DESC 
    LIMIT 5
    ''').fetchall()
    conn.close()

    return render_template('backup.html',
                           backup_devices=backup_devices,
                           recent_records=recent_records)


@app.route('/api/backup', methods=['POST'])
@login_required
def create_backup():
    """创建数据备份"""
    try:
        # 获取备份路径
        backup_path = request.json.get('backup_path')
        if not backup_path:
            return {'success': False, 'error': '请选择备份路径'}

        # 创建备份
        result = perform_backup(backup_path)

        if result['success']:
            flash('数据备份成功！', 'success')
            return {'success': True, 'message': '备份完成', 'backup_file': result['backup_file']}
        else:
            return {'success': False, 'error': result['error']}

    except Exception as e:
        return {'success': False, 'error': f'备份失败: {str(e)}'}


@app.route('/api/restore', methods=['POST'])
@login_required
def restore_backup():
    """恢复数据备份"""
    try:
        backup_file = request.files.get('backup_file')
        if not backup_file:
            return {'success': False, 'error': '请选择备份文件'}

        # 执行恢复
        result = perform_restore(backup_file)

        if result['success']:
            flash('数据恢复成功！', 'success')
            return {'success': True, 'message': '恢复完成'}
        else:
            return {'success': False, 'error': result['error']}

    except Exception as e:
        return {'success': False, 'error': f'恢复失败: {str(e)}'}

# 用户管理路由

@app.route('/admin/users')
@login_required
def user_management():
    """用户管理页面 - 只有管理员可访问"""
    if session.get('user_role') != 'admin':
        flash('权限不足！只有管理员可以访问用户管理页面。', 'error')
        return redirect(url_for('index'))

    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    conn.close()

    users_list = []
    for user in users:
        users_list.append(dict(user))

    return render_template('user_management.html', users=users_list)


@app.route('/admin/users/add', methods=['POST'])
@login_required
def add_user():
    """添加用户"""
    if session.get('user_role') != 'admin':
        return {'success': False, 'error': '权限不足'}, 403

    try:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        real_name = request.form.get('real_name', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', 'user')

        if not username or not password:
            return {'success': False, 'error': '用户名和密码不能为空'}

        if len(password) < 6:
            return {'success': False, 'error': '密码长度至少6位'}

        conn = get_db_connection()

        # 检查用户名是否已存在
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ?', (username,)
        ).fetchone()

        if existing_user:
            conn.close()
            return {'success': False, 'error': '用户名已存在'}

        # 添加新用户
        conn.execute(
            'INSERT INTO users (username, password_hash, role, real_name, phone) VALUES (?, ?, ?, ?, ?)',
            (username, hash_password(password), role, real_name, phone)
        )

        conn.commit()
        conn.close()

        flash(f'用户 {username} 添加成功！', 'success')
        return {'success': True, 'message': '用户添加成功'}

    except Exception as e:
        return {'success': False, 'error': f'添加用户失败: {str(e)}'}


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    """删除用户"""
    if session.get('user_role') != 'admin':
        return {'success': False, 'error': '权限不足'}, 403

    try:
        # 防止删除自己
        if user_id == session.get('user_id'):
            return {'success': False, 'error': '不能删除自己的账户'}

        conn = get_db_connection()

        # 获取用户信息（用于提示）
        user = conn.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()

        if not user:
            conn.close()
            return {'success': False, 'error': '用户不存在'}

        # 删除用户
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()

        flash(f'用户 {user["username"]} 已删除！', 'success')
        return {'success': True, 'message': '用户删除成功'}

    except Exception as e:
        return {'success': False, 'error': f'删除用户失败: {str(e)}'}


@app.route('/admin/users/<int:user_id>/reset_password', methods=['POST'])
@login_required
def reset_user_password(user_id):
    """重置用户密码"""
    if session.get('user_role') != 'admin':
        return {'success': False, 'error': '权限不足'}, 403

    try:
        new_password = request.form.get('new_password', '').strip()

        if not new_password:
            return {'success': False, 'error': '新密码不能为空'}

        if len(new_password) < 6:
            return {'success': False, 'error': '密码长度至少6位'}

        conn = get_db_connection()

        # 获取用户信息
        user = conn.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()

        if not user:
            conn.close()
            return {'success': False, 'error': '用户不存在'}

        # 重置密码
        conn.execute(
            'UPDATE users SET password_hash = ? WHERE id = ?',
            (hash_password(new_password), user_id)
        )

        conn.commit()
        conn.close()

        flash(f'用户 {user["username"]} 的密码已重置！', 'success')
        return {'success': True, 'message': '密码重置成功'}

    except Exception as e:
        return {'success': False, 'error': f'重置密码失败: {str(e)}'}


# 备份记录查询路由
@app.route('/admin/backup_records')
@login_required
def backup_records():
    """查看备份记录 - 管理员功能"""
    # 检查权限
    if session.get('user_role') != 'admin':
        flash('只有管理员可以查看备份记录', 'error')
        return redirect(url_for('index'))

    conn = get_db_connection()

    # 查询备份记录
    records = conn.execute('''
    SELECT 
        br.id,
        br.user_id,
        br.backup_path,
        br.backup_type,
        br.backup_size,
        br.status,
        br.notes,
        br.created_at,
        u.username,
        u.real_name
    FROM backup_records br
    LEFT JOIN users u ON br.user_id = u.id
    ORDER BY br.created_at DESC
    ''').fetchall()

    conn.close()

    # 转换为字典列表
    records_list = []
    for record in records:
        record_dict = dict(record)
        records_list.append(record_dict)

    return render_template('admin/backup_records.html', records=records_list)

def reextract_all_features():
    """重新提取数据库中所有图片的特征向量"""
    conn = get_db_connection()
    patterns = conn.execute('SELECT id, image_path, name FROM patterns').fetchall()

    updated_count = 0
    for pattern in patterns:
        pattern_id = pattern['id']
        image_path = pattern['image_path']
        pattern_name = pattern['name']
        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)

        if os.path.exists(full_image_path):
            print(f"重新提取特征: {pattern_name} ({image_path})")
            new_features = extract_features(full_image_path)

            # 更新数据库
            feature_blob = pickle.dumps(new_features)
            conn.execute(
                'UPDATE patterns SET feature_vector = ? WHERE id = ?',
                (feature_blob, pattern_id)
            )
            updated_count += 1
            print(f"  已更新 {pattern_name} 的特征向量")
        else:
            print(f"图片文件不存在: {image_path}")

    conn.commit()
    conn.close()
    print(f"特征向量更新完成，共更新 {updated_count} 条记录")
    return updated_count


if __name__ == '__main__':
    # 初始化数据库
    init_db()

    print("=" * 50)
    print("苗绣纹样管理系统启动成功！")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)

    # 自动打开浏览器（只在主进程中执行）
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()

    # 启动Flask应用
    app.run(debug=False, host='0.0.0.0', port=5000)  # 关闭debug模式