---
layout: compress
permalink: '/:path/mindmap.js'
---

/* 脑图查看页 —— Mermaid mindmap 渲染 + 「更多 >>」系列弹窗
 *
 * 两部分彼此独立，互不影响：
 *
 *   A. mindmap —— 用 Mermaid 的 render() API 生成脑图 SVG，
 *      再把节点与站内文章链接绑定，并按分族着色。
 *      主题的 page.min.js 只处理 ```mermaid 代码块（.language-mermaid），
 *      而本页用自己的容器（.mm-diagram），所以这里自行调用。
 *
 *   B. series modal —— 「更多 >>」弹窗。
 *      主题自带的 Bootstrap 只打包了 collapse / tooltip 的 data-api，
 *      没有 Modal 组件也没有对应样式，因此这里用原生实现：
 *      遮罩层 + 固定定位面板 + 手动管理焦点与滚动锁定。
 */

/* ================================================================
 * A. Mermaid mindmap
 * ================================================================ */
(() => {
  'use strict';

  const root = document.querySelector('.mindmap-page');
  if (!root) return;

  const host = root.querySelector('.mm-diagram');
  const srcEl = root.querySelector('.mm-src');
  if (!host || !srcEl) return;

  const source = (srcEl.textContent || '').trim();
  if (!source) return;

  const SVG_NS = 'http://www.w3.org/2000/svg';

  /* ---------------------------------------------------------- 数据源 */

  /* 节点文字 → 文章链接。键为家谱条目里的 key（与脑图节点文字完全一致）。 */
  const NODE_LINKS = {
    {%- for fam in site.data.optimization_family.families -%}
      {%- for m in fam.members %}
    '{{ m.key | escape }}': {
      url: '{{ m.url | relative_url }}',
      family: '{{ fam.id }}',
      color: '{{ fam.color }}',
      abbr: '{{ m.abbr | escape }}',
      title: '{{ m.title | escape }}'
    },
      {%- endfor %}
    {%- endfor %}
  };

  /* 分族名 → 颜色，用于给族节点上色 */
  const FAMILY_COLORS = {
    {%- for fam in site.data.optimization_family.families %}
    '{{ fam.name | escape }}': '{{ fam.color }}',
    {%- endfor %}
  };

  /* ------------------------------------------------------- 节点处理 */

  /** 取出节点的纯文本标签 */
  function nodeText(node) {
    const label = node.querySelector('.nodeLabel') || node.querySelector('g.label');
    const raw = label ? (label.textContent || '') : (node.textContent || '');
    // Mermaid mindmap 的引号会被当作字面量，这里统一去掉首尾引号
    return raw.replace(/\s+/g, ' ').trim().replace(/^["']|["']$/g, '').trim();
  }

  /** 去掉末尾的英文缩写，得到纯中文名 */
  function stripAbbr(label, info) {
    if (info && info.abbr && label.endsWith(' ' + info.abbr)) {
      return label.slice(0, -(info.abbr.length + 1));
    }
    return label;
  }

  /**
   * 用 SVG 的 <a> 包住节点。
   * 节点 <g> 自带 transform，<a> 不带 transform，因此包裹后位置不变。
   */
  function wrapInLink(node, url, ariaLabel) {
    const parent = node.parentNode;
    if (!parent) return;

    // 已经包过就跳过
    if (node.parentNode && node.parentNode.tagName.toLowerCase() === 'a') return;

    const a = document.createElementNS(SVG_NS, 'a');
    a.setAttribute('href', url);
    a.setAttribute('target', '_blank');
    a.setAttribute('rel', 'noopener noreferrer');
    a.setAttribute('aria-label', ariaLabel);
    a.setAttribute('class', 'mm-node__link');

    parent.insertBefore(a, node);
    a.appendChild(node);
  }

  /** 给节点加链接与配色 */
  function decorate(svg) {
    const nodes = svg.querySelectorAll('g.node, g.nodeWithId');
    if (!nodes.length) return;

    nodes.forEach((node) => {
      const label = nodeText(node);
      if (!label) return;

      const info = NODE_LINKS[label];
      const familyColor = FAMILY_COLORS[label];

      if (info) {
        node.classList.add('mm-node');
        node.style.setProperty('--mm-node-color', info.color);
        wrapInLink(node, info.url, `${stripAbbr(label, info)} — ${info.abbr}`);
        return;
      }

      // 分族节点：按族名上色
      if (familyColor) {
        node.classList.add('mm-node', 'mm-node--family');
        node.style.setProperty('--mm-node-color', familyColor);
      }
    });
  }

  /* ------------------------------------------------------- 渲染流程 */

  function currentTheme() {
    const mode = document.documentElement.getAttribute('data-mode');
    if (mode === 'dark' || mode === 'light') return mode;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'default';
  }

  async function render() {
    if (typeof mermaid === 'undefined' || typeof mermaid.render !== 'function') return;

    try {
      mermaid.initialize({
        startOnLoad: false,
        theme: currentTheme(),
        securityLevel: 'loose',
        fontFamily: 'inherit',
        mindmap: { padding: 16 }
      });

      const id = 'mm-mindmap-' + Date.now();
      const { svg } = await mermaid.render(id, source);

      host.innerHTML = svg;
      const svgEl = host.querySelector('svg');
      if (svgEl) {
        svgEl.removeAttribute('width');
        svgEl.removeAttribute('height');
        svgEl.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        decorate(svgEl);
      }
    } catch (err) {
      console.error('mindmap: 渲染失败', err);
      host.innerHTML =
        '<p class="mm-canvas__error">脑图渲染失败，请刷新页面重试；下方家谱明细不受影响。</p>';
    }
  }

  /* Mermaid 库是 defer 脚本，可能晚于本脚本执行，做一次就绪检测 */
  let tries = 0;
  const timer = setInterval(() => {
    if (typeof mermaid !== 'undefined') {
      clearInterval(timer);
      render();
      return;
    }
    tries += 1;
    if (tries > 80) clearInterval(timer);
  }, 100);

  /* --------------------------------------------------- 主题切换重绘 */

  window.addEventListener('message', (e) => {
    if (!e.data || e.data.id !== 'theme-mode') return;
    // 主题脚本切换后 postMessage，这里等 DOM 更新再重绘
    setTimeout(render, 150);
  });
})();

/* ================================================================
 * B. 页面交互：「更多 >>」系列弹窗 + 卡片展开
 * ================================================================ */
(() => {
  'use strict';

  const root = document.querySelector('.mindmap-page');
  if (!root) return;

  /* ------------------------------------------------ 卡片内展开/收起 */

  root.querySelectorAll('.mm-family__toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const card = btn.closest('.mm-family');
      if (!card) return;

      const expanded = card.classList.toggle('is-expanded');
      btn.setAttribute('aria-expanded', String(expanded));

      const text = btn.querySelector('.mm-family__toggle-text');
      if (text) {
        const extra = card.querySelectorAll('.mm-member--extra').length;
        text.textContent = expanded
          ? `收起其余 ${extra} 个`
          : `展开其余 ${extra} 个`;
      }
    });
  });

  /* ------------------------------------------------ 「更多 >>」弹窗 */

  const modal = document.getElementById('mm-series-modal');
  const moreBtn = document.getElementById('mm-more-btn');
  if (!modal || !moreBtn) return;

  let lastFocused = null;
  let closeTimer = null;

  /** 切换到某个系列面板 */
  function selectSeries(id) {
    modal.querySelectorAll('.mm-series-tab').forEach((tab) => {
      const on = tab.dataset.series === id;
      tab.classList.toggle('is-active', on);
      tab.setAttribute('aria-selected', String(on));
      tab.setAttribute('tabindex', on ? '0' : '-1');
    });

    modal.querySelectorAll('.mm-series-panel').forEach((panel) => {
      panel.classList.toggle('is-active', panel.id === `mm-panel-${id}`);
    });
  }

  function openModal() {
    lastFocused = document.activeElement;
    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
    modal.hidden = false;
    // 触发进场动画
    requestAnimationFrame(() => modal.classList.add('is-open'));
    document.documentElement.classList.add('mm-modal-open');

    const first = modal.querySelector('.mm-series-tab.is-active') || modal.querySelector('.mm-series-tab');
    if (first) first.focus();
  }

  function closeModal() {
    modal.classList.remove('is-open');
    document.documentElement.classList.remove('mm-modal-open');

    if (closeTimer) clearTimeout(closeTimer);
    closeTimer = setTimeout(() => {
      modal.hidden = true;
      closeTimer = null;
    }, 220);

    if (lastFocused && typeof lastFocused.focus === 'function') lastFocused.focus();
  }

  moreBtn.addEventListener('click', openModal);

  // 点击遮罩、关闭按钮、关闭按钮组都能关闭
  modal.querySelectorAll('[data-mm-close]').forEach((el) => {
    el.addEventListener('click', closeModal);
  });

  // 系列标签切换
  modal.querySelectorAll('.mm-series-tab').forEach((tab) => {
    tab.addEventListener('click', () => selectSeries(tab.dataset.series));
  });

  // Esc 关闭，左右方向键切换系列
  document.addEventListener('keydown', (e) => {
    if (modal.hidden) return;

    if (e.key === 'Escape') {
      closeModal();
      return;
    }
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;

    const tabs = Array.from(modal.querySelectorAll('.mm-series-tab'));
    const idx = tabs.indexOf(document.activeElement);
    if (idx === -1) return;

    e.preventDefault();
    const next = (idx + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectSeries(tabs[next].dataset.series);
    tabs[next].focus();
  });
})();
