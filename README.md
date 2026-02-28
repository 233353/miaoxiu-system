# 苗绣纹样智能管理系统
 
**项目定位**：非遗文化数字化管理平台，支持纹样录入、分类检索、权限管理  
**在线演示**：http://47.97.101.124  
**项目源码**：https://github.com/233353/miaoxiu-system
> 
> 2.**联系作者**：1844657767@qq.com
> 

## 部署文档
> 1.服务器准备：
> 系统：Alibaba Cloud Linux 3/ CentOS 7+
> 配置：2核2G及以上
> 安全组：开放80端口
>
> 2.环境安装：
```bash
# 更新系统
yum update -y

# 安装Python、Nginx
yum install python3 python3.11-pip nginx -y

# 启动Nginx
systemctl start nginx
systemctl enable nginx
```
> 下载WinSCP
> 3.项目部署：
```bash
# 上传项目到 /var/www/miaoxiu
cd /var/www/miaoxiu

# 安装依赖
pip3 install -r requirements.txt

# 设置环境变量
export SECRET_KEY="your-production-secret-key"
echo 'export SECRET_KEY="your-production-secret-key"' >> ~/.bashrc

# 初始化管理员密码
python3.11 reset_admin.py

# 启动应用程序
nohup python3.11 app.py > app.log 2>&1 &
```

> 4.Nginx配置
```bash
server{
    listen 80;
    server_name 你的IP或域名;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
   }

   location /static{
       alias /var/www/miaoxiu/static;
   }
}
```

> 5.重启Nginx
```bash
nginx -t
systemctl restart nginx
```

## 项目亮点
| 特性           | 说明 | 技术实现 |
|--------------|------|----------|
| **AI纹样识别** | 基于ResNet50提取特征向量 | PaddlePaddle深度学习框架 |
| **智能相似搜索**   | 余弦相似度算法匹配 | NumPy向量计算 |
| **色彩分析**     | K-means聚类提取主色调 | scikit-learn机器学习 |
| **权限管理**     | 多角色用户系统 | Flask-Login + Session |
| **数据安全**     | 完整备份恢复机制 | ZIP压缩 + 文件管理 |

## 功能模块

### 1. 用户管理模块
- **多角色系统**：管理员/普通用户
- **安全认证**：Werkzeug密码哈希加密
- **用户操作**：注册、登录、修改信息

### 2. 纹样管理模块
- **上传功能**：支持JPG/PNG格式，自动特征提取
- **编辑功能**：修改纹样信息（名称、地区、传承人等）
- **检索功能**：基于AI的相似纹样搜索
- **分析功能**：色彩提取、纹样对比

### 3. 系统管理模块
- **数据备份**：一键备份数据库和图片
- **数据恢复**：从备份文件恢复系统
- **操作日志**：记录用户操作和系统事件

## 技术架构
- 后端：Flask + SQLite + PaddlePaddle
- 前端：Bootstrap 5 + jQuery
- AI：ResNet50 + 余弦相似度算法

## 快速开始
```bash
# 环境要求
- Python 3.11+
- SQLite 3
- 至少2GB内存（用于AI模型）

# 安装步骤

# 1. 克隆项目
git clone https://github.com/233353/miaoxiu-system.git
cd miaoxiu-project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 初始化数据库
python reset_admin.py

# 4. 启动应用
python app.py

