# reset_admin_github.py - 这个上传到GitHub
import sqlite3
import os
from werkzeug.security import generate_password_hash


def reset_admin():
    print("=" * 50)
    print("苗绣纹样管理系统 - 数据库初始化")
    print("=" * 50)

    # 安全提示
    print("安全提示：请先设置管理员密码")
    print("")
    print("使用方法：")
    print("1. 打开本文件，找到第24行左右")
    print("2. 修改 MY_PASSWORD 变量的值")
    print("3. 保存文件后重新运行")
    print("")
    print("示例：")
    print('   MY_PASSWORD = "YourStrongPassword123!"')
    print("=" * 50)

    MY_PASSWORD = input("请输入管理员密码: ").strip()

    print("正在初始化数据库...")

    # 备份旧数据库
    if os.path.exists('database.db'):
        os.remove('database.db')
        print("已删除旧数据库")

    # 创建新数据库
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # 建表
    cursor.execute('''
        CREATE TABLE users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            real_name TEXT,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    cursor.execute('''
            CREATE TABLE patterns(
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

    cursor.execute('''
            CREATE TABLE backup_records(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                backup_path VARCHAR(500) NOT NULL,
                backup_type VARCHAR(50) NOT NULL,
                backup_size BIGINT,
                status VARCHAR(20) NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

    # 创建或更新管理员账户
    password_hash = generate_password_hash(MY_PASSWORD)
    cursor.execute(
        '''INSERT OR REPLACE INTO users (username, password_hash, role, real_name) 
           VALUES (?, ?, ?, ?)''',
        ('admin', password_hash, 'admin', '系统管理员')
    )

    conn.commit()
    conn.close()

    print("=" * 40)
    print(" 数据库初始化完成！")
    print(f"管理员账号: admin")
    print(f"密码: {MY_PASSWORD}")
    print("请记住这个密码！")
    print("=" * 40)
    print("下一步：运行 python app.py 启动系统")
    print("然后访问 http://localhost:5000 登录")


if __name__ == '__main__':
    reset_admin()