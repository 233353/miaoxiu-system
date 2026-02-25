import sqlite3
import hashlib


def check_database_state():
    """检查数据库当前状态"""
    print("检查数据库状态...")
    print("=" * 60)

    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("数据库中的表：")
        for table in tables:
            print(f"  - {table[0]}")

        print("\n用户数据：")
        cursor.execute("SELECT id, username, role, real_name, created_at FROM users")
        users = cursor.fetchall()

        if not users:
            print(" 用户表为空！")
        else:
            for user in users:
                print(f"  ID:{user[0]} 用户:{user[1]:10s} 角色:{user[2]:6s} 姓名:{user[3]:10s} 创建时间:{user[4]}")

        # 检查默认密码
        print("\n 默认密码验证：")
        test_password = 'password123'
        expected_hash = hashlib.md5(test_password.encode()).hexdigest()

        cursor.execute("SELECT username, password_hash FROM users WHERE username='admin'")
        admin = cursor.fetchone()

        if admin:
            if admin[1] == expected_hash:
                print(f"   admin账户密码是默认值: {test_password}")
            else:
                print(f"   admin密码已被修改")
                print(f"     当前哈希: {admin[1]}")
                print(f"     默认哈希: {expected_hash}")
        else:
            print("   admin账户不存在！")

        # 检查纹样数据
        print("\n  纹样数据统计：")
        cursor.execute("SELECT COUNT(*) FROM patterns")
        pattern_count = cursor.fetchone()[0]
        print(f"  纹样记录数: {pattern_count}")

        conn.close()

    except Exception as e:
        print(f"数据库检查失败: {e}")


def show_login_info():
    """显示登录信息"""
    print("\n" + "=" * 60)
    print(" 当前可用的登录账户：")
    print("-" * 60)

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT username, role, real_name FROM users")
    users = cursor.fetchall()

    if users:
        for username, role, real_name in users:
            if username == 'admin':
                print(f" 管理员账户: {username} / password123")
                print(f"   姓名: {real_name}")
            else:
                print(f" 普通账户: {username} / password123")
                print(f"   角色: {role}, 姓名: {real_name}")
            print()
    else:
        print(" 数据库中没有用户账户")

    conn.close()


if __name__ == "__main__":
    check_database_state()
    show_login_info()

    print("\n💡 提示：")
    print("1. 如果显示默认密码，请使用 admin / password123 登录")
    print("2. 如果想修改密码，运行: python reset_password.py")
    print("3. 如果想清除所有数据重新开始，直接删除 database.db 文件")