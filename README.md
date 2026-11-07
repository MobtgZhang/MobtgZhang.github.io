# MobtgZhang.github.io

> 简单的程序员 —— 个人博客与知识库

这是 [MobtgZhang](https://github.com/mobtgzhang) 的个人站点，托管在 GitHub Pages 上：
**<https://mobtgzhang.github.io>**

站点基于 [Jekyll](https://jekyllrb.com/) 与 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 主题构建，
在主题之上做了较多定制：智能优化算法家谱脑图、实时开源仓库页、全站动态形象与天气插件等。

[![Jekyll](https://img.shields.io/badge/Jekyll-4.4-red.svg)](https://jekyllrb.com/)
[![Jekyll Theme Chirpy](https://img.shields.io/badge/theme-Chirpy-7.5-blue.svg)](https://github.com/cotes2020/jekyll-theme-chirpy)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 目录

- [站点内容](#站点内容)
- [特色功能](#特色功能)
- [技术栈](#技术栈)
- [本地运行](#本地运行)
- [目录结构](#目录结构)
- [写作规范](#写作规范)
- [部署](#部署)
- [致谢与许可](#致谢与许可)

---

## 站点内容

当前共 **25 篇**文章，主要覆盖两个方向：

| 分类 | 数量 | 说明 |
| :--- | :---: | :--- |
| 智能计算 | 17 | 智能优化算法详解，含数学原理、伪代码与应用 |
| 论文 | 2 | 翻译模型、知识表征学习方向的论文笔记 |
| 翻译模型 | 2 | Encoder–Decoder、注意力机制与高斯变换建模 |
| 阅读理解 | 1 | R-Net 机器阅读理解模型 |
| 评价方法 | 1 | MRR / Hit@10 / ACC / ROC / AUC / DCG / NDCG |
| 数据集 | 1 | 中文机器阅读理解数据集整理 |
| 关系抽取论文 | 1 | 实体关系抽取论文合集 |

智能优化算法按搜索机制分为四族，收录 17 个算法：

- **进化计算** —— ES、GA、DE、CMA-ES
- **群智能** —— ACO、PSO、ABC、FA、CS、BA、WOA、GWO、HHO、SSA
- **教学与谐振** —— HS、TLBO
- **经典基准** —— SA（模拟退火）

每篇文章都标注了原始文献出处与发表年份，站内 `/mindmap-viewer/` 脑图页可按机制跳转对应文章。

---

## 特色功能

除 Chirpy 主题自带能力外，本站实现了以下定制功能：

### 算法家谱脑图（`/mindmap-viewer/`）

用 [Mermaid](https://mermaid.js.org/) 的 mindmap 语法渲染算法家谱，
并通过 `assets/js/mindmap.js` 做后处理：把 SVG 节点与站内文章链接绑定、按分族着色。
「更多 >>」弹窗内可切换学习路线 / NLP 地图 / 全站索引三个系列。

> 主题自带的 Bootstrap 仅打包了 Tooltip 与 Collapse 的 data-api，没有 Modal 组件，
> 因此该弹窗为原生实现（遮罩层 + 焦点管理 + 滚动锁定）。

### 开源页实时数据（`/open-source/`）

构建期由 `_plugins/github_public_repos.rb` 抓取一次 GitHub API 作为静态兜底，
页面加载后 `assets/js/open-source.js` 再用访客自己的 API 额度拉取实时数据覆盖。
**不做任何本地缓存**，每次访问都是最新数据。

### 天气插件（页面右上角）

液态玻璃风格卡片，含当前实况、近 24 小时与近 7 天预报。
图标采用 [Meteocons](https://github.com/basmilius/meteocons)（MIT），
玻璃质感与交互改编自 [weatherwidget](https://github.com/weatherwidget/weatherwidget.github.io)（MIT）。

定位优先级：浏览器 Geolocation → IP 定位（ipwho.is）→ 站点默认城市。
数据来自 [Open-Meteo](https://open-meteo.com)，免 API Key、支持 CORS。

### 全站动态形象（右下角）

基于 [webmeji](https://github.com/lars-rooij/webmeji)，在每个页面右下角展示动态漫画形象。
可通过 `_config.yml` 的 `webmeji.pages` 配置仅在指定页面启用。

### 其它

- 全站 MathJax 支持（单页可在 front matter 设 `math: false` 关闭）
- PWA 可安装与离线缓存
- 明暗主题切换，脑图随主题自动重绘

### 读书（`/books/`）与宝典（`/handbook/`）

两个栏目都是**纯数据驱动**的，页面本身不含内容：

| 栏目 | 数据源 | 说明 |
| :--- | :--- | :--- |
| 读书 | `_data/books.yml` | 按 `field`（计算机 / 数学 / 物理）分组，支持状态、评分、标签与 Markdown 读书笔记 |
| 宝典 | `_data/handbook.yml` | 按 `sections` 分板块，支持外部链接、标签与代码片段（面试题速查） |

新增记录只需往对应 YAML 里追加条目，页面会自动出现对应的卡片 / 板块；
删掉条目即恢复为占位空态。布局分别见 `_layouts/books.html` 与 `_layouts/handbook.html`。

> **注意**：数学公式里不要直接写 `|`（会被 kramdown 当成表格分隔符），
> 请用 `\lvert` / `\rvert` 代替。

---

## 技术栈

| 类别 | 选型 |
| :--- | :--- |
| 站点生成 | Jekyll 4.4.1 |
| 主题 | jekyll-theme-chirpy 7.5.0 |
| 语言 / 插件 | Ruby 3.3、Bundler 2.6 |
| 部署 | GitHub Actions → GitHub Pages |
| 图表 | Mermaid 11.13.0 |
| 数学公式 | MathJax 3 |
| 天气数据 | Open-Meteo API |
| 仓库数据 | GitHub REST API |

第三方静态资源全部内置在 `assets/lib/`，不依赖运行时 CDN；版本记录在
`assets/lib/package.json` 与 `assets/lib/README.md`，可用 `assets/lib/update.sh` 更新。

---

## 本地运行

```shell
# 安装依赖（vendor/bundle）
bundle install

# 启动本地服务器，默认 http://127.0.0.1:4000
bundle exec jekyll serve

# 生产环境构建（压缩 CSS/HTML）
JEKYLL_ENV=production bundle exec jekyll build
```

> **注意**：GitHub 默认的 `github-pages` gem 是 Jekyll 3，与本项目使用的 Jekyll 4 不兼容，
> 因此部署走 GitHub Actions，而非 Jekyll 官方 GitHub Pages 构建流程。

若构建时 GitHub API 返回 403（未认证请求限流），终端会给出提示并回退到静态兜底数据；
设置环境变量 `JEKYLL_GITHUB_TOKEN` 可提高构建期限额。

---

## 目录结构

```text
.
├── _config.yml              # 站点配置
├── _data/                   # 数据源（YAML）
│   ├── optimization_family.yml   # 算法家谱：分族与成员元数据
│   ├── mindmap_series.yml        # 脑图「更多 >>」弹窗的系列内容
│   ├── open_source_projects.yml  # 开源页静态兜底数据
│   └── locales/zh-CN.yml         # 中文文案
├── _includes/               # 主题片段（含 weather / webmeji）
├── _layouts/                # 自定义布局
│   ├── mindmap-viewer.html      # 算法家谱页
│   ├── open-source.html         # 开源页
│   ├── books.html               # 读书页（/_data/books.yml 驱动）
│   └── handbook.html            # 宝典页（/_data/handbook.yml 驱动）
├── _plugins/                # Ruby 插件
│   ├── github_public_repos.rb   # 构建期抓取 GitHub 仓库数据
│   └── posts-lastmod-hook.rb    # 用 git 时间戳补 lastmod
├── _posts/                  # 文章（25 篇）
├── _sass/custom.scss        # 自定义样式
├── _tabs/                   # 导航页（关于 / 开源 / 脑图查看 等）
├── _wiki/                   # 维基条目（黎曼猜想系列）
├── assets/
│   ├── js/                  # 自定义脚本
│   └── lib/                 # 内置的第三方静态资源
└── .github/workflows/       # GitHub Actions 部署
```

---

## 写作规范

- 文章置于 `_posts/`，文件名格式 `YYYY-MM-DD-Title.md`，永久链接为 `/posts/:title/`
- 每篇文章需在 front matter 中声明 `categories` 与 `tags`
- 智能优化算法类文章需标注**原始文献出处与发表年份**
- `last_modified_at` 由 `_plugins/posts-lastmod-hook.rb` 从 git 提交时间自动推导，无需手写

---

## 部署

推送到 `main` 分支即触发 `.github/workflows/pages.yml`：

1. Checkout → 配置 Ruby 3.3 → 自检 Jekyll 环境
2. `bundle exec jekyll build`（`JEKYLL_ENV=production`）
3. 上传 `_site` 并部署到 GitHub Pages

也支持在 Actions 面板手动触发（`workflow_dispatch`）。

### ⚠️ 首次部署须知：关闭 GitHub 自动生成的旧工作流

`github.io` 仓库在首次启用 Pages 时，GitHub 会自动生成一个名为
**「Deploy Jekyll with GitHub Pages dependencies preinstalled」** 的工作流，
它调用 `actions/jekyll-build-pages@v1`。该 action **不安装你的 `Gemfile`**，
而是使用镜像内置的 `github-pages` gem（Jekyll 3.10），因此对本项目会报：

```
The jekyll-theme-chirpy theme could not be found.
(Jekyll::Errors::MissingDependencyException)
```

它只是在 `bundle check` 不通过时打印一条 warning，然后**继续用 Jekyll 3 构建**并失败。

**处理办法**（在仓库 **Settings → Pages → Build and deployment → Source**
中把 Source 设为 **GitHub Actions**，并到 **Actions** 标签页禁用/删除那个自动生成的
`Deploy Jekyll with GitHub Pages dependencies preinstalled` 工作流）。

仓库中已提交的 `.github/workflows/pages.yml` 才是正确的工作流；
只要上面那个自动生成的工作流还在，它就仍会在 push 时被触发并失败。

---

## 致谢与许可

- 主题：[jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)（MIT）
- 天气插件图标：[Meteocons](https://github.com/basmilius/meteocons)（MIT）
- 玻璃质感交互：[weatherwidget](https://github.com/weatherwidget/weatherwidget.github.io)（MIT）
- 动态形象：[webmeji](https://github.com/lars-rooij/webmeji)
- 静态资源：[chirpy-static-assets](https://github.com/cotes2020/chirpy-static-assets)

本仓库代码以 [MIT](LICENSE) 许可发布；`LICENSE` 中的版权声明继承自 Chirpy Starter 模板。
