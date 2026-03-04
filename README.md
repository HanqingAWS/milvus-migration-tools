# Milvus Migration Tools

开源 Milvus 向量数据库数据迁移工具集，支持将 Collection 数据（Schema + 索引 + 全量数据）从一个 Milvus 实例导出并导入到另一个 Milvus 实例。

适用于 Milvus 2.4.x 开源版，支持跨主机、跨环境的数据迁移。

## 功能特性

- **数据导出** (`export_data.py`)：将 Collection 的 Schema、索引配置、全量数据导出为 JSON 文件
- **数据导入** (`import_data.py`)：从 JSON 文件中读取数据，在目标 Milvus 实例上重建 Collection 并导入
- **迁移验证** (`verify_migration.py`)：对比源和目标实例的数据一致性（条数、Schema、抽样、搜索结果）

## 适用场景

| 场景 | 说明 |
|------|------|
| 主机迁移 | 将 Milvus 从一台服务器迁移到另一台 |
| 环境迁移 | 从测试环境迁移到生产环境 |
| 云迁移 | 在不同云厂商或区域间迁移 |
| 版本升级 | 配合 Milvus 版本升级进行数据迁移 |
| 数据备份 | 将数据导出为 JSON 文件进行离线备份 |

> **注意**：本工具适用于百万级以下数据集。对于更大规模的数据，建议使用 [milvus-backup](https://github.com/zilliztech/milvus-backup) 官方工具。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 从源 Milvus 导出数据

```bash
python3 export_data.py \
    --host 10.0.1.100 \
    --port 19530 \
    --collection my_collection \
    --output ./milvus_export
```

导出产物：

```
milvus_export/
├── schema.json        # Collection Schema 定义
├── index.json         # 索引配置（类型、参数）
├── data.json          # 全量数据（含向量和标量字段）
└── export_meta.json   # 导出元数据（时间戳、记录数等）
```

### 3. 导入到目标 Milvus

```bash
python3 import_data.py \
    --host 10.0.2.200 \
    --port 19530 \
    --input ./milvus_export \
    --drop-existing
```

### 4. 验证迁移结果

```bash
python3 verify_migration.py \
    --source-host 10.0.1.100 --source-port 19530 \
    --target-host 10.0.2.200 --target-port 19530 \
    --collection my_collection \
    --output verify_results.json
```

验证项包括：

| # | 验证内容 | 说明 |
|---|---------|------|
| 1 | Collection 存在性 | 源和目标实例上都存在该 Collection |
| 2 | 数据条数一致 | 两端记录数完全相同 |
| 3 | Schema 匹配 | 字段名称和数据类型完全一致 |
| 4 | 抽样比对 | 前 10 条记录的标量字段和向量数据一致 |
| 5 | 随机抽查 | 随机 20 条记录的完整比对 |
| 6 | 搜索一致性 | 相同 query 向量的 Top-10 结果 ID 和分数一致 |

## 命令参考

### export_data.py

```
usage: export_data.py [-h] --collection COLLECTION [--host HOST] [--port PORT]
                      [--user USER] [--password PASSWORD]
                      [--output OUTPUT] [--batch-size BATCH_SIZE]

参数说明：
  --host            Milvus 地址（默认: localhost）
  --port            Milvus 端口（默认: 19530）
  --user            用户名（开启认证时使用）
  --password        密码（开启认证时使用）
  --collection      要导出的 Collection 名称（必填）
  --output          导出目录（默认: ./milvus_export）
  --batch-size      查询批次大小（默认: 500）
```

### import_data.py

```
usage: import_data.py [-h] --input INPUT [--host HOST] [--port PORT]
                      [--user USER] [--password PASSWORD]
                      [--batch-size BATCH_SIZE] [--drop-existing]

参数说明：
  --host            Milvus 地址（默认: localhost）
  --port            Milvus 端口（默认: 19530）
  --user            用户名（开启认证时使用）
  --password        密码（开启认证时使用）
  --input           导出文件所在目录（必填）
  --batch-size      插入批次大小（默认: 200）
  --drop-existing   如果目标已存在同名 Collection，先删除再导入
```

### verify_migration.py

```
usage: verify_migration.py [-h] --source-host SOURCE_HOST --target-host TARGET_HOST
                           --collection COLLECTION
                           [--source-port SOURCE_PORT] [--target-port TARGET_PORT]
                           [--user USER] [--password PASSWORD]
                           [--output OUTPUT] [--spot-check-count N]

参数说明：
  --source-host         源 Milvus 地址（必填）
  --source-port         源 Milvus 端口（默认: 19530）
  --target-host         目标 Milvus 地址（必填）
  --target-port         目标 Milvus 端口（默认: 19530）
  --user                用户名
  --password            密码
  --collection          要验证的 Collection 名称（必填）
  --output              验证结果 JSON 输出路径
  --spot-check-count    随机抽查数量（默认: 20）
```

## 完整迁移示例

以下示例演示将游戏客服知识库从旧主机 `10.0.1.100` 迁移到新主机 `10.0.2.200`：

```bash
# Step 1: 在新主机上部署 Milvus（Docker Compose 方式）
ssh user@10.0.2.200
mkdir -p ~/milvus && cd ~/milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.4.17/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker compose up -d

# Step 2: 在可以访问两台主机的机器上安装工具
git clone https://github.com/HanqingAWS/milvus-migration-tools.git
cd milvus-migration-tools
pip install -r requirements.txt

# Step 3: 从旧主机导出
python3 export_data.py \
    --host 10.0.1.100 \
    --port 19530 \
    --collection game_cs_knowledge \
    --output ./milvus_export

# Step 4: 导入到新主机
python3 import_data.py \
    --host 10.0.2.200 \
    --port 19530 \
    --input ./milvus_export \
    --drop-existing

# Step 5: 验证
python3 verify_migration.py \
    --source-host 10.0.1.100 \
    --target-host 10.0.2.200 \
    --collection game_cs_knowledge \
    --output verify_results.json

# Step 6: 更新应用配置（仅需修改连接地址）
# 将应用中的 MILVUS_HOST 从 10.0.1.100 改为 10.0.2.200
# 重启应用服务即可，无需修改任何业务代码
```

## 迁移后应用代码需要改什么？

**只需要修改 Milvus 的连接地址（Host/IP），其他代码完全不用改。**

```python
# 迁移前
connections.connect(host="10.0.1.100", port="19530")

# 迁移后 —— 仅改这一行
connections.connect(host="10.0.2.200", port="19530")
```

以下均无需修改：

- Collection 名称
- 字段定义 / Schema
- 索引类型和参数
- Insert / Search / Query 逻辑
- RAG Pipeline 流程
- Embedding 模型配置

## 支持的数据类型

| 数据类型 | 支持 |
|---------|:----:|
| INT8 / INT16 / INT32 / INT64 | Yes |
| FLOAT / DOUBLE | Yes |
| BOOL | Yes |
| VARCHAR | Yes |
| JSON | Yes |
| FLOAT_VECTOR | Yes |
| BINARY_VECTOR | Yes |

## 环境要求

- Python 3.8+
- pymilvus 2.4.x
- numpy
- 源和目标 Milvus 版本：2.4.x

## License

MIT
