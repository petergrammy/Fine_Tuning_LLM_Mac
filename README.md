## FT_llm：用个人文本做 LoRA 微调（基于 LlamaFactory）

本目录已经完成：

- **拉取并安装**：`LlamaFactory` 已克隆到 `./LlamaFactory`，并在 `./.venv`（Python 3.12）中完成可编辑安装 + `requirements/metrics.txt`。
- **接下来要做**：把你的个人文本放进 `raw/`，运行转换脚本生成数据集，然后用配置文件启动 LoRA 训练。

### 1) 激活环境

```bash
source ./.venv/bin/activate
python -V
```

### 2) 准备你的个人文本

把原始文本放到 `raw/` 目录（你可以放多个 `.txt` 文件）。建议：

- 同一文件尽量保持同一写作风格（比如朋友圈/邮件/日记分开）。
- 先去掉明显的隐私信息（手机号、地址、身份证号等）。

目录结构示例：

```text
raw/
  diary.txt
  posts.txt
```

### 3) 生成数据集（两种路线）

我们提供两种最常用路线，你可以先跑通 **CPT（继续预训练）**，再做 **SFT（指令微调）**。

#### A. CPT（继续预训练，最贴近“文风”）

把文本切成片段，生成 `jsonl`：每行一个 `{"text": ...}`。

```bash
python tools/prepare_personal_dataset.py \
  --input_dir raw \
  --output_dir personal_data \
  --mode cpt \
  --dataset_name personal_cpt \
  --chunk_chars 1200
```

#### B. SFT（指令微调，更像“对话”）

会把你的文本当作“回答”，用固定提示词构造简单的指令数据（适合快速得到“像你说话”的聊天效果，但质量高度依赖数据清洗）。

```bash
python tools/prepare_personal_dataset.py \
  --input_dir raw \
  --output_dir personal_data \
  --mode sft \
  --dataset_name personal_sft \
  --chunk_chars 1200
```

### 4) 开始训练（先跑通 sanity check）

先用极小模型跑通流程（CPU/MPS 都能跑，但会慢；主要用于验证配置与数据集没问题）：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m llamafactory.cli train configs/train_sanity_lora_sft.yaml
```

训练成功后，再把 `configs/` 里模型换成你想用的（如 Qwen2.5/3、Llama3 等），并适当调大 `cutoff_len`、`max_steps/num_train_epochs` 等。

### 5) 推理（加载 LoRA）

训练完成后，在 `output_dir` 下会得到 adapter。你可以用：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m llamafactory.cli chat configs/infer_sanity_lora.yaml
```

> 备注：如果你有代理环境变量导致 Hugging Face 报 `ProxyError: 403 Forbidden`，上面的 `env -u ...` 会临时取消代理。
> 若你遇到下载限速/403（非代理），可设置 `HF_TOKEN`（Hugging Face 访问令牌）提升限额。

### Phi-3 额外提示

如果你使用 `microsoft/Phi-3-mini-4k-instruct`，**不要开启 `trust_remote_code`**，否则可能遇到 `KeyError: 'type'`（rope_scaling 字段不兼容）的问题。

### 几点提示

1.本项目使用QWen作为base是最容易跑通的，但是效果并不理想，原因很简单：大陆的AI经过了非常傻逼的法律限制，基本没办法锐评任何东西，所以建议用Phi作为base去跑
2.使用方法是将你自己的文本粘贴在text文件夹里的文件中，不过如果你只是想和一个说话像peter的llm聊天的话也可以不用替换文本喵
3.learn bash
4.除非你对自己的mac性能很自信，否则**不要在mac上跑**，很容易出现一个叫做7.12GiB空间限制的报错
