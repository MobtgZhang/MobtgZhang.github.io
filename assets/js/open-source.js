---
layout: compress
permalink: '/:path/open-source.js'
---

/* 开源页 —— 实时数据刷新
 *
 * 页面在构建期已经渲染出静态兜底数据；本脚本在浏览器端再次向 GitHub API
 * 拉取该用户的公开仓库，用访客自己的 API 额度把 star / fork / 语言 /
 * 更新时间等数字刷新成实时值，并在顶部汇总出总览统计。
 *
 * GitHub REST API 返回 Access-Control-Allow-Origin: *，浏览器可直接跨域请求。
 * 未认证请求的限额为 60 次/小时/IP，因此不做任何本地缓存 ——
 * 每次访问页面都直接向 GitHub API 请求最新数据。
 * 请求失败时保留构建期渲染的静态兜底数据，页面依然可读。
 *
 * 数据来源：https://api.github.com/users/{username}/repos
 */

(() => {
  'use strict';

  const USERNAME = '{{ site.github.username | default: "mobtgzhang" }}';
  const API = `https://api.github.com/users/${USERNAME}/repos?per_page=100&sort=updated`;

  const table = document.getElementById('os-table');
  if (!table) return;

  const searchInput = document.getElementById('os-search');
  const langFilter = document.getElementById('os-lang-filter');
  const emptyHint = document.getElementById('os-empty');
  const moreBtn = document.getElementById('os-more');
  const statRepos = document.getElementById('os-stat-repos');
  const statStars = document.getElementById('os-stat-stars');
  const statForks = document.getElementById('os-stat-forks');
  const statLangs = document.getElementById('os-stat-langs');
  const statUpdated = document.getElementById('os-stat-updated');

  /* 「全部仓库」默认只显示最近更新的这几个，可展开全部 */
  const RECENT_LIMIT = 8;
  /* 精选项目固定为主页那 6 个仓库，不随 API 结果增减 */
  const FEATURED_NAMES = [
    'sues-thesis',
    'CMRC2018',
    'VHD-Boot-Olds',
    'zh_latex_journals',
    '2021MathorCupBigDataCompetition',
    'sues-mm'
  ];

  /* 语言 → 主题色点（与 GitHub Linguist 常用色一致） */
  const LANG_COLORS = {
    'Python': '#3572A5', 'JavaScript': '#f1e05a', 'TypeScript': '#3178c6',
    'Java': '#b07219', 'C': '#555555', 'C++': '#f34b7d', 'C#': '#178600',
    'Go': '#00ADD8', 'Rust': '#dea584', 'Ruby': '#701516', 'PHP': '#4F5D95',
    'Swift': '#F05138', 'Kotlin': '#A97BFF', 'Dart': '#00B4AB', 'Scala': '#c22d40',
    'Shell': '#89e051', 'HTML': '#e34c26', 'CSS': '#563d7c', 'Vue': '#41b883',
    'Jupyter Notebook': '#DA5B0B', 'TeX': '#3D6117', 'Lua': '#000080',
    'Zig': '#ec915c', 'Elixir': '#6e4a7e', 'Haskell': '#5e5086', 'Clojure': '#db5855',
    'Erlang': '#B83998', 'OCaml': '#3be133', 'F#': '#b845fc', 'Nim': '#ffc200',
    'R': '#198CE7', 'MATLAB': '#e16737', 'Groovy': '#4298b8', 'Crystal': '#000100'
  };

  const langColor = (lang) => LANG_COLORS[lang] || '#8b949e';

  /* ---------------------------------------------------------- 工具函数 */

  /** 把 ISO 时间转成「相对多久之前」，例如 3 天前 */
  function relativeTime(iso) {
    if (!iso) return '—';
    const t = new Date(iso).getTime();
    if (Number.isNaN(t)) return '—';

    const diff = Date.now() - t;
    const min = 60 * 1000;
    const hour = 60 * min;
    const day = 24 * hour;

    if (diff < min) return '刚刚';
    if (diff < hour) return `${Math.floor(diff / min)} 分钟前`;
    if (diff < day) return `${Math.floor(diff / hour)} 小时前`;
    if (diff < 30 * day) return `${Math.floor(diff / day)} 天前`;
    if (diff < 365 * day) return `${Math.floor(diff / (30 * day))} 个月前`;
    return `${Math.floor(diff / (365 * day))} 年前`;
  }

  function escapeHtml(str) {
    return String(str ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  /* ------------------------------------------------------- 渲染：列表 */

  function buildRow(repo) {
    const a = document.createElement('a');
    a.className = 'os-row os-row--repo';
    a.href = repo.html_url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.setAttribute('role', 'row');
    a.dataset.repo = repo.name;
    a.dataset.lang = repo.language || 'other';
    a.dataset.stars = repo.stargazers_count || 0;
    a.dataset.forks = repo.forks_count || 0;
    a.dataset.updated = repo.updated_at || '';
    a.dataset.description = repo.description || '';

    a.innerHTML = `
      <span class="os-row__name" role="cell">
        <i class="fas fa-book-open" aria-hidden="true"></i>
        <span class="os-row__label">
          <b>${escapeHtml(repo.name)}</b>
          <small>${escapeHtml(repo.description || '')}</small>
        </span>
      </span>
      <span class="os-row__lang" role="cell">
        <span class="os-dot" style="background:${langColor(repo.language)}"></span>${escapeHtml(repo.language || '—')}
      </span>
      <span class="os-row__num" role="cell"><i class="fas fa-star" aria-hidden="true"></i>${repo.stargazers_count || 0}</span>
      <span class="os-row__num" role="cell"><i class="fas fa-code-branch" aria-hidden="true"></i>${repo.forks_count || 0}</span>
      <span class="os-row__time" role="cell">${relativeTime(repo.updated_at)}</span>
    `;
    return a;
  }

  /** 用实时数据重建整个列表：按最近更新时间倒序，最近的排最前 */
  function renderList(repos) {
    // 移除静态兜底行，保留表头
    table.querySelectorAll('.os-row--repo').forEach((r) => r.remove());

    const sorted = [...repos].sort((a, b) => {
      const ta = Date.parse(a.pushed_at || a.updated_at || 0) || 0;
      const tb = Date.parse(b.pushed_at || b.updated_at || 0) || 0;
      return tb - ta;
    });

    const frag = document.createDocumentFragment();
    sorted.forEach((repo, i) => {
      const row = buildRow(repo);
      // 最近的几个默认展示，其余折叠
      if (i >= RECENT_LIMIT) row.classList.add('is-collapsed');
      frag.appendChild(row);
    });
    table.appendChild(frag);

    syncMoreButton();
    applyFilters();
  }

  /* --------------------------------------------------- 展开 / 收起按钮 */

  function syncMoreButton() {
    if (!moreBtn) return;
    const collapsed = [...table.querySelectorAll('.os-row--repo.is-collapsed')];
    const count = collapsed.length;
    const text = moreBtn.querySelector('.os-more__text');
    const badge = moreBtn.querySelector('.os-more__count');

    if (!count) {
      moreBtn.hidden = true;
      return;
    }
    moreBtn.hidden = false;
    if (badge) badge.textContent = count;

    const expanded = moreBtn.getAttribute('aria-expanded') === 'true';
    const label = expanded
      ? moreBtn.dataset.lessLabel || '收起'
      : moreBtn.dataset.moreLabel || '展开全部仓库';
    if (text) text.textContent = label;
  }

  moreBtn?.addEventListener('click', () => {
    const expanded = moreBtn.getAttribute('aria-expanded') === 'true';
    moreBtn.setAttribute('aria-expanded', String(!expanded));
    table.querySelectorAll('.os-row--repo.is-collapsed').forEach((r) => {
      r.classList.toggle('is-collapsed', expanded);
    });
    syncMoreButton();
  });

  /* --------------------------------------------------- 渲染：顶部统计 */

  function renderStats(repos) {
    const stars = repos.reduce((s, r) => s + (r.stargazers_count || 0), 0);
    const forks = repos.reduce((s, r) => s + (r.forks_count || 0), 0);
    const langs = new Set(repos.map((r) => r.language).filter(Boolean));

    if (statRepos) statRepos.textContent = repos.length;
    if (statStars) statStars.textContent = stars;
    if (statForks) statForks.textContent = forks;
    if (statLangs) statLangs.textContent = langs.size;

    // 最近更新 = 最新 push 时间
    const latest = repos
      .map((r) => r.pushed_at || r.updated_at)
      .filter(Boolean)
      .sort()
      .pop();
    if (statUpdated) statUpdated.textContent = latest ? relativeTime(latest) : '—';

    // 同步精选卡片的数字
    repos.forEach((repo) => {
      const card = document.querySelector(`.os-feature[data-repo="${CSS.escape(repo.name)}"]`);
      if (!card) return;

      const star = card.querySelector('[data-star]');
      const fork = card.querySelector('[data-fork]');
      const updated = card.querySelector('[data-updated]');
      const badge = card.querySelector('[data-lang-badge]');

      if (star) star.textContent = repo.stargazers_count || 0;
      if (fork) fork.textContent = repo.forks_count || 0;
      if (updated) updated.textContent = relativeTime(repo.pushed_at || repo.updated_at);
      if (badge && repo.language) {
        badge.dataset.langBadge = repo.language;
        badge.innerHTML = `<span class="os-dot" style="background:${langColor(repo.language)}"></span>${escapeHtml(repo.language)}`;
      }
    });
  }

  /* --------------------------------------------------- 语言筛选按钮 */

  function renderLangFilter(repos) {
    if (!langFilter) return;

    const counts = new Map();
    repos.forEach((r) => {
      if (!r.language) return;
      counts.set(r.language, (counts.get(r.language) || 0) + 1);
    });

    const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);

    langFilter.innerHTML = '';
    const all = document.createElement('button');
    all.type = 'button';
    all.className = 'os-lang-btn is-active';
    all.dataset.lang = '';
    all.textContent = '全部';
    all.addEventListener('click', () => {
      langFilter.querySelectorAll('.os-lang-btn').forEach((b) => b.classList.remove('is-active'));
      all.classList.add('is-active');
      activeLang = '';
      applyFilters();
    });
    langFilter.appendChild(all);

    sorted.forEach(([lang, count]) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'os-lang-btn';
      btn.dataset.lang = lang;
      btn.innerHTML = `<span class="os-dot" style="background:${langColor(lang)}"></span>${escapeHtml(lang)} <b>${count}</b>`;
      btn.addEventListener('click', () => {
        langFilter.querySelectorAll('.os-lang-btn').forEach((b) => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        activeLang = lang;
        applyFilters();
      });
      langFilter.appendChild(btn);
    });
  }

  /* --------------------------------------------------------- 筛选逻辑 */

  let activeLang = '';

  function applyFilters() {
    const keyword = (searchInput?.value || '').trim().toLowerCase();
    // 正在搜索时把折叠的仓库也一并展示出来，方便找到目标
    const searching = keyword.length > 0;

    let visibleIndex = 0;
    table.querySelectorAll('.os-row--repo').forEach((row) => {
      const matchLang = !activeLang || row.dataset.lang === activeLang;
      const haystack = `${row.dataset.repo} ${row.dataset.description}`.toLowerCase();
      const matchText = !keyword || haystack.includes(keyword);
      const visible = matchLang && matchText;

      row.hidden = !visible;
      // 用 hidden 属性时，行内 flex 布局会失效，这里用 class 控制
      row.classList.toggle('is-hidden', !visible);

      if (visible) {
        visibleIndex += 1;
        // 搜索时全部展开；正常状态下只有最近更新的前几个保持展开
        row.classList.toggle('is-collapsed', !searching && visibleIndex > RECENT_LIMIT);
      }
    });

    const anyVisible = visibleIndex > 0;
    if (emptyHint) emptyHint.hidden = anyVisible;
  }

  searchInput?.addEventListener('input', applyFilters);

  /* --------------------------------------------------- 拉取实时数据 */

  /** 从页面已有的行里读出静态兜底数据，用于 API 不可用时填充统计数字 */
  function readStaticRows() {
    return [...table.querySelectorAll('.os-row--repo')].map((row) => ({
      name: row.dataset.repo,
      html_url: row.href,
      description: row.dataset.description || '',
      language: row.dataset.lang && row.dataset.lang !== 'other' ? row.dataset.lang : null,
      stargazers_count: Number(row.dataset.stars) || 0,
      forks_count: Number(row.dataset.forks) || 0,
      updated_at: row.dataset.updated || null,
      pushed_at: row.dataset.updated || null
    }));
  }

  async function loadLive() {
    try {
      // 每次访问都直接请求 GitHub API，不做任何本地缓存
      const res = await fetch(API, {
        headers: { Accept: 'application/vnd.github+json' },
        cache: 'no-store'
      });

      if (!res.ok) throw new Error(`GitHub API ${res.status}`);

      const repos = await res.json();
      if (!Array.isArray(repos) || repos.length === 0) return;

      // 过滤掉 fork（作者的 fork 往往是别人的项目）
      const own = repos.filter((r) => !r.fork);
      const list = own.length > 0 ? own : repos;

      renderStats(list);
      renderList(list);
      renderLangFilter(list);

      const note = document.querySelector('.os-source-note');
      if (note) {
        note.classList.remove('os-source-note--warn');
        note.classList.add('is-live');
      }
    } catch (err) {
      // 请求失败：用页面里已有的静态数据把统计数字补齐，页面依然完整可读
      console.warn('open-source: 无法获取实时数据，显示静态数据', err.message);
      const statics = readStaticRows();
      if (statics.length) {
        renderStats(statics);
        renderLangFilter(statics);
      }
    }
  }

  loadLive();
})();
