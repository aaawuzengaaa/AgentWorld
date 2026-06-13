import pandas as pd
import json
import os

def csv_to_json(csv_file_path, json_file_path):
    """
    将CSV文件转换为JSON格式
    """
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        
        print(f"成功读取 {len(df)} 行数据")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {df.columns.tolist()}")
        
        # 处理NaN值，将其转换为None
        df = df.where(pd.notnull(df), None)
        
        # 转换为字典列表
        data_list = df.to_dict('records')
        
        # 保存为JSON文件
        print(f"正在保存为JSON文件: {json_file_path}")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成! JSON文件已保存到: {json_file_path}")
        print(f"JSON文件大小: {os.path.getsize(json_file_path) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False

def main():
    # 文件路径
    csv_file = "Workflow_Evaluation_Processed.csv"
    json_file = "Workflow_Evaluation_Processed.json"
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 找不到CSV文件 {csv_file}")
        return
    
    # 执行转换
    success = csv_to_json(csv_file, json_file)
    
    if success:
        print("\n=== 转换摘要 ===")
        # 读取一小部分JSON来展示结构
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"总记录数: {len(data)}")
            if data:
                print("第一条记录的字段:")
                for key in data[0].keys():
                    print(f"  - {key}")

if __name__ == "__main__":
    main() 