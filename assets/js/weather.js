/* 天气插件 —— 页面右上角的液态玻璃天气卡片
 *
 * 图标：Meteocons (https://github.com/basmilius/meteocons)，MIT License，
 *       来自 @meteocons/svg 的 fill 风格，内含 SMIL 动画。
 * 玻璃质感与交互：Adapted from
 *       https://github.com/weatherwidget/weatherwidget.github.io (MIT License).
 *
 * 定位优先级：
 *   1. 浏览器 Geolocation API（最精确，需用户授权）
 *   2. IP 定位（ipwho.is，免 key，基于访客出口 IP）
 *   3. 站点默认城市（构建期配置）
 *
 * 数据来源：Open-Meteo（https://open-meteo.com），免 API Key、支持 CORS。
 *   - current  : 当前实况
 *   - hourly   : 近 24 小时（past_hours=24）
 *   - daily    : 近 7 天（past_days=7）+ 未来 7 天（forecast_days=7）
 * 结果缓存 30 分钟于 sessionStorage，避免每次切页重复请求。
 */

(() => {
  'use strict';

  const ROOT_ID = 'weather-plugin';
  const CACHE_KEY = 'site:weather-cache';
  const CACHE_TTL = 30 * 60 * 1000;  // 30 分钟
  const GEOLOCATE_TIMEOUT = 8000;    // 8 秒后放弃高精度定位
  const IP_TIMEOUT = 6000;
  const ICON_BASE = '/assets/img/weather/';

  /* WMO weather code → 描述（双语）、场景类型与 Meteocons 图标
   *
   * icon      : 白天的图标（nightIcon 用于夜间）
   * scene     : 驱动玻璃卡片的渐变主题
   */
  const WMO = {
    0:  { scene: 'sunny',         icon: 'clear-day',          nightIcon: 'clear-night',          en: 'Clear sky',           zh: '晴朗' },
    1:  { scene: 'sunny',         icon: 'mostly-clear-day',   nightIcon: 'mostly-clear-night',   en: 'Mainly clear',        zh: '大致晴朗' },
    2:  { scene: 'partly-cloudy', icon: 'partly-cloudy-day',  nightIcon: 'partly-cloudy-night',  en: 'Partly cloudy',       zh: '局部多云' },
    3:  { scene: 'overcast',      icon: 'overcast-day',       nightIcon: 'overcast-night',       en: 'Overcast',            zh: '阴天' },
    45: { scene: 'misty',         icon: 'fog-day',            nightIcon: 'fog-night',            en: 'Foggy',               zh: '有雾' },
    48: { scene: 'misty',         icon: 'fog-day',            nightIcon: 'fog-night',            en: 'Depositing rime fog', zh: '雾凇' },
    51: { scene: 'rainy',         icon: 'drizzle',            nightIcon: 'drizzle',              en: 'Light drizzle',       zh: '小毛毛雨' },
    53: { scene: 'rainy',         icon: 'drizzle',            nightIcon: 'drizzle',              en: 'Moderate drizzle',    zh: '毛毛雨' },
    55: { scene: 'rainy',         icon: 'drizzle',            nightIcon: 'drizzle',              en: 'Dense drizzle',       zh: '浓毛毛雨' },
    56: { scene: 'rainy',         icon: 'sleet',              nightIcon: 'sleet',                en: 'Freezing drizzle',    zh: '冻毛毛雨' },
    57: { scene: 'rainy',         icon: 'sleet',              nightIcon: 'sleet',                en: 'Heavy freezing drizzle', zh: '强冻毛毛雨' },
    61: { scene: 'rainy',         icon: 'rain',               nightIcon: 'rain',                 en: 'Slight rain',         zh: '小雨' },
    63: { scene: 'rainy',         icon: 'rain',               nightIcon: 'rain',                 en: 'Moderate rain',       zh: '中雨' },
    65: { scene: 'rainy',         icon: 'rain',               nightIcon: 'rain',                 en: 'Heavy rain',          zh: '大雨' },
    66: { scene: 'rainy',         icon: 'sleet',              nightIcon: 'sleet',                en: 'Freezing rain',       zh: '冻雨' },
    67: { scene: 'rainy',         icon: 'sleet',              nightIcon: 'sleet',                en: 'Heavy freezing rain', zh: '强冻雨' },
    71: { scene: 'snowy',         icon: 'snow',               nightIcon: 'snow',                 en: 'Slight snow',         zh: '小雪' },
    73: { scene: 'snowy',         icon: 'snow',               nightIcon: 'snow',                 en: 'Moderate snow',       zh: '中雪' },
    75: { scene: 'snowy',         icon: 'snow',               nightIcon: 'snow',                 en: 'Heavy snow',          zh: '大雪' },
    77: { scene: 'snowy',         icon: 'snow',               nightIcon: 'snow',                 en: 'Snow grains',         zh: '雪粒' },
    80: { scene: 'rainy',         icon: 'rain',               nightIcon: 'rain',                 en: 'Slight rain showers', zh: '阵雨' },
    81: { scene: 'rainy',         icon: 'rain',               nightIcon: 'rain',                 en: 'Moderate rain showers', zh: '强阵雨' },
    82: { scene: 'stormy',        icon: 'rain',               nightIcon: 'rain',                 en: 'Violent rain showers', zh: '暴雨' },
    85: { scene: 'snowy',         icon: 'snow',               nightIcon: 'snow',                 en: 'Slight snow showers', zh: '阵雪' },
    86: { scene: 'snowy',         icon: 'snow',               nightIcon: 'snow',                 en: 'Heavy snow showers',  zh: '强阵雪' },
    95: { scene: 'stormy',        icon: 'thunderstorms-day',  nightIcon: 'thunderstorms-night',  en: 'Thunderstorm',        zh: '雷阵雨' },
    96: { scene: 'stormy',        icon: 'hail',               nightIcon: 'hail',                 en: 'Thunderstorm with hail', zh: '雷阵雨伴冰雹' },
    99: { scene: 'stormy',        icon: 'hail',               nightIcon: 'hail',                 en: 'Thunderstorm with heavy hail', zh: '强雷暴伴冰雹' }
  };

  const FALLBACK = { scene: 'overcast', icon: 'cloudy', nightIcon: 'cloudy', en: 'Unknown', zh: '未知天气' };

  /* UI 文案（双语）。默认英文，中文可一键切换。 */
  const I18N = {
    en: {
      humidity: 'Humidity', wind: 'Wind', clouds: 'Clouds',
      locate: 'Use my location', refresh: 'Refresh',
      loading: 'Fetching weather…', locating: 'Locating you…',
      error: 'Failed to load weather', feelsLike: 'Feels like',
      toggle: 'Toggle weather widget',
      hourly: 'Past 24 hours', daily: 'Last 7 days',
      noData: 'No data'
    },
    zh: {
      humidity: '湿度', wind: '风速', clouds: '云量',
      locate: '使用我的位置', refresh: '刷新',
      loading: '正在获取天气…', locating: '正在定位…',
      error: '天气加载失败', feelsLike: '体感',
      toggle: '切换天气插件',
      hourly: '近 24 小时', daily: '近 7 天',
      noData: '暂无数据'
    }
  };

  const root = document.getElementById(ROOT_ID);
  if (!root) return;

  const el = (id) => document.getElementById(id);

  const toggleBtn = el('weather-plugin-toggle');
  const panel = el('weather-plugin-panel');
  const glass = root.querySelector('.weather-plugin__glass');
  const loading = el('weather-plugin-loading');
  const loadingText = el('weather-plugin-loading-text');
  const content = el('weather-plugin-content');
  const errorBox = el('weather-plugin-error');
  const particlesHost = el('weather-plugin-particles');
  const refreshBtn = el('weather-plugin-refresh');
  const locateBtn = el('weather-plugin-locate');
  const toggleIcon = el('weather-plugin-toggle-icon');
  const toggleTemp = el('weather-plugin-toggle-temp');
  const hourlyBody = el('weather-plugin-hourly-body');
  const dailyBody = el('weather-plugin-daily-body');

  const defaultLat = parseFloat(root.dataset.defaultLat) || 31.2304;
  const defaultLon = parseFloat(root.dataset.defaultLon) || 121.4737;
  const defaultCity = root.dataset.defaultCity || '上海';

  /* 语言：默认英文，记忆用户选择 */
  const LANG_KEY = 'site:weather-lang';
  let lang = 'en';
  try {
    lang = localStorage.getItem(LANG_KEY) === 'zh' ? 'zh' : 'en';
  } catch (_) { /* 隐私模式下不可用 */ }

  const t = (key) => I18N[lang][key] || I18N.en[key] || key;

  /* ---------------------------------------------------------------- 定位 */

  function locateByBrowser() {
    return new Promise((resolve) => {
      if (!navigator.geolocation) return resolve(null);

      let settled = false;
      const done = (v) => { if (!settled) { settled = true; resolve(v); } };

      navigator.geolocation.getCurrentPosition(
        (pos) => done({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          city: null,
          source: 'gps'
        }),
        () => done(null),
        { timeout: GEOLOCATE_TIMEOUT, maximumAge: 10 * 60 * 1000, enableHighAccuracy: false }
      );
    });
  }

  async function locateByIp() {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), IP_TIMEOUT);
      const res = await fetch('https://ipwho.is/', { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) return null;

      const data = await res.json();
      if (!data || data.success === false) return null;
      if (typeof data.latitude !== 'number' || typeof data.longitude !== 'number') return null;

      return {
        lat: data.latitude,
        lon: data.longitude,
        city: data.city || data.region || data.country || null,
        source: 'ip'
      };
    } catch (_) {
      return null;
    }
  }

  /* ------------------------------------------------------------ 天气数据 */

  async function fetchWeather(lat, lon) {
    const params = new URLSearchParams({
      latitude: lat.toFixed(4),
      longitude: lon.toFixed(4),
      current: 'temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,is_day',
      hourly: 'temperature_2m,weather_code,precipitation_probability',
      daily: 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset',
      past_hours: '24',
      past_days: '7',
      forecast_days: '7',
      timezone: 'auto'
    });

    const res = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
    if (!res.ok) throw new Error(`weather api ${res.status}`);

    const data = await res.json();
    if (!data || !data.current) throw new Error('weather payload invalid');
    return data;
  }

  /* -------------------------------------------------------------- 渲染 */

  function describe(code) {
    return WMO[code] || FALLBACK;
  }

  function iconName(code, isDay) {
    const info = describe(code);
    const name = isDay ? info.icon : (info.nightIcon || info.icon);
    return `${ICON_BASE}${name}.svg`;
  }

  function windDirection(deg) {
    if (typeof deg !== 'number') return '';
    const dirs = lang === 'zh'
      ? ['北', '东北', '东', '东南', '南', '西南', '西', '西北']
      : ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    return dirs[Math.round(((deg % 360) / 45)) % 8];
  }

  /* 背景漂浮粒子 */
  function buildParticles() {
    if (!particlesHost) return;
    particlesHost.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 14; i++) {
      const p = document.createElement('span');
      p.className = 'weather-plugin__particle';
      p.style.left = `${Math.random() * 100}%`;
      p.style.bottom = `${Math.random() * 60}%`;
      p.style.animationDelay = `${(Math.random() * 8).toFixed(2)}s`;
      p.style.animationDuration = `${(6 + Math.random() * 8).toFixed(2)}s`;
      p.style.width = p.style.height = `${(2 + Math.random() * 4).toFixed(1)}px`;
      frag.appendChild(p);
    }
    particlesHost.appendChild(frag);
  }

  function applyStaticText() {
    // 切换语言时更新所有带 data-i18n 的静态文案
    root.querySelectorAll('[data-i18n]').forEach((node) => {
      const key = node.dataset.i18n;
      if (I18N[lang][key]) node.textContent = I18N[lang][key];
    });
    toggleBtn?.setAttribute('aria-label', t('toggle'));
    if (loadingText && loading && !loading.hidden) loadingText.textContent = t('loading');
    if (errorBox && !errorBox.hidden && errorBox.dataset.messageKey) {
      errorBox.textContent = t(errorBox.dataset.messageKey);
    }
  }

  /* ------------------------------------------------- 近 24 小时（小时级） */

  function renderHourly(hourly) {
    if (!hourlyBody || !hourly || !Array.isArray(hourly.time)) {
      if (hourlyBody) hourlyBody.innerHTML = '';
      return;
    }

    // 找出「当前小时」在序列中的位置：取最后一个小于等于当前时间的点
    const now = Date.now();
    let startIdx = 0;
    for (let i = 0; i < hourly.time.length; i++) {
      if (new Date(hourly.time[i]).getTime() <= now) startIdx = i;
      else break;
    }

    const items = [];
    for (let i = startIdx; i < hourly.time.length && items.length < 24; i++) {
      const ts = new Date(hourly.time[i]);
      items.push({
        time: ts,
        label: ts.toLocaleTimeString(lang === 'zh' ? 'zh-CN' : 'en-US', { hour: '2-digit', minute: '2-digit', hour12: lang === 'en' }),
        temp: hourly.temperature_2m ? hourly.temperature_2m[i] : null,
        code: hourly.weather_code ? hourly.weather_code[i] : null,
        pop: hourly.precipitation_probability ? hourly.precipitation_probability[i] : null
      });
    }

    if (!items.length) {
      hourlyBody.innerHTML = `<p class="weather-plugin__empty">${t('noData')}</p>`;
      return;
    }

    const frag = document.createDocumentFragment();
    items.forEach((it, idx) => {
      const cell = document.createElement('div');
      cell.className = 'weather-plugin__hour';
      if (idx === 0) cell.classList.add('is-now');

      const icon = document.createElement('img');
      icon.className = 'weather-plugin__hour-icon';
      icon.src = iconName(it.code, true);
      icon.alt = describe(it.code)[lang] || describe(it.code).en;
      icon.loading = 'lazy';
      icon.width = 28;
      icon.height = 28;

      const time = document.createElement('span');
      time.className = 'weather-plugin__hour-time';
      time.textContent = idx === 0 && lang === 'en' ? 'Now' : it.label;

      const temp = document.createElement('span');
      temp.className = 'weather-plugin__hour-temp';
      temp.textContent = it.temp == null ? '--' : `${Math.round(it.temp)}°`;

      const pop = document.createElement('span');
      pop.className = 'weather-plugin__hour-pop';
      if (it.pop != null && it.pop > 0) {
        pop.textContent = `${it.pop}%`;
      }

      cell.append(icon, time, temp, pop);
      frag.appendChild(cell);
    });

    hourlyBody.innerHTML = '';
    hourlyBody.appendChild(frag);
  }

  /* --------------------------------------------------- 近 7 天（日级） */

  function renderDaily(daily) {
    if (!dailyBody || !daily || !Array.isArray(daily.time)) {
      if (dailyBody) dailyBody.innerHTML = '';
      return;
    }

    const now = new Date();
    now.setHours(0, 0, 0, 0);

    // 取今天以及之前的 7 天（含今天，共 7 天）
    const items = [];
    for (let i = 0; i < daily.time.length; i++) {
      const ts = new Date(daily.time[i]);
      ts.setHours(0, 0, 0, 0);
      if (ts <= now) {
        items.push({
          date: ts,
          code: daily.weather_code ? daily.weather_code[i] : null,
          max: daily.temperature_2m_max ? daily.temperature_2m_max[i] : null,
          min: daily.temperature_2m_min ? daily.temperature_2m_min[i] : null,
          pop: daily.precipitation_probability_max ? daily.precipitation_probability_max[i] : null,
          sunrise: daily.sunrise ? daily.sunrise[i] : null,
          sunset: daily.sunset ? daily.sunset[i] : null
        });
      }
    }
    items.reverse();  // 最近的排在前面

    // 「近一周」= 今天 + 之前 6 天，共 7 天
    const week = items.slice(0, 7);

    if (!week.length) {
      dailyBody.innerHTML = `<p class="weather-plugin__empty">${t('noData')}</p>`;
      return;
    }

    const frag = document.createDocumentFragment();
    week.forEach((it, idx) => {
      const row = document.createElement('div');
      row.className = 'weather-plugin__day';
      if (idx === 0) row.classList.add('is-today');

      const day = document.createElement('span');
      day.className = 'weather-plugin__day-name';
      if (idx === 0) {
        day.textContent = lang === 'zh' ? '今天' : 'Today';
      } else {
        day.textContent = it.date.toLocaleDateString(lang === 'zh' ? 'zh-CN' : 'en-US', { weekday: 'short' });
      }

      const icon = document.createElement('img');
      icon.className = 'weather-plugin__day-icon';
      icon.src = iconName(it.code, true);
      icon.alt = describe(it.code)[lang] || describe(it.code).en;
      icon.loading = 'lazy';
      icon.width = 26;
      icon.height = 26;

      const pop = document.createElement('span');
      pop.className = 'weather-plugin__day-pop';
      if (it.pop != null && it.pop > 0) pop.textContent = `${it.pop}%`;

      const range = document.createElement('span');
      range.className = 'weather-plugin__day-range';
      const max = it.max == null ? '--' : `${Math.round(it.max)}°`;
      const min = it.min == null ? '--' : `${Math.round(it.min)}°`;
      range.innerHTML = `<b>${max}</b><i>${min}</i>`;

      row.append(day, icon, pop, range);
      frag.appendChild(row);
    });

    dailyBody.innerHTML = '';
    dailyBody.appendChild(frag);
  }

  /* ------------------------------------------------------------ 主渲染 */

  function render(data, city) {
    const current = data.current;
    const info = describe(current.weather_code);
    const isNight = current.is_day === 0;
    const scene = info.scene;

    // 场景类型驱动 CSS 渐变
    root.dataset.scene = scene;
    if (glass) glass.dataset.scene = scene;

    el('weather-plugin-temp').textContent = Math.round(current.temperature_2m);
    el('weather-plugin-unit').textContent = '°C';
    el('weather-plugin-desc').textContent = info[lang] || info.en;
    el('weather-plugin-feels').textContent = `${t('feelsLike')} ${Math.round(current.apparent_temperature)}°C`;
    el('weather-plugin-humidity').textContent = `${current.relative_humidity_2m}%`;
    el('weather-plugin-clouds').textContent = `${current.cloud_cover}%`;

    const dir = windDirection(current.wind_direction_10m);
    el('weather-plugin-wind').textContent = `${Math.round(current.wind_speed_10m)} km/h${dir ? ` ${dir}` : ''}`;

    el('weather-plugin-city').textContent = city || defaultCity;

    // 主图标（Meteocons 动态 SVG）
    const mainIcon = el('weather-plugin-icon');
    if (mainIcon) {
      mainIcon.innerHTML = '';
      const img = document.createElement('img');
      img.className = 'weather-plugin__icon-img';
      img.src = iconName(current.weather_code, !isNight);
      img.alt = info[lang] || info.en;
      img.width = 72;
      img.height = 72;
      mainIcon.appendChild(img);
    }

    // 折叠按钮上的小图标
    toggleIcon.innerHTML = '';
    const small = document.createElement('img');
    small.className = 'weather-plugin__toggle-img';
    small.src = iconName(current.weather_code, !isNight);
    small.alt = '';
    small.width = 20;
    small.height = 20;
    toggleIcon.appendChild(small);
    toggleTemp.textContent = `${Math.round(current.temperature_2m)}°`;

    // 历史数据
    renderHourly(data.hourly);
    renderDaily(data.daily);

    applyStaticText();

    loading.hidden = true;
    errorBox.hidden = true;
    content.hidden = false;
    root.dataset.state = 'ready';
  }

  function showError(key) {
    root.dataset.state = 'error';
    loading.hidden = true;
    content.hidden = true;
    errorBox.hidden = false;
    errorBox.dataset.messageKey = key;
    errorBox.textContent = t(key);
  }

  /* -------------------------------------------------------------- 缓存 */

  function readCache() {
    try {
      const raw = sessionStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      const cached = JSON.parse(raw);
      if (!cached || typeof cached.ts !== 'number') return null;
      if (Date.now() - cached.ts > CACHE_TTL) return null;
      return cached;
    } catch (_) {
      return null;
    }
  }

  function writeCache(payload) {
    try {
      sessionStorage.setItem(CACHE_KEY, JSON.stringify({ ...payload, ts: Date.now() }));
    } catch (_) { /* ignore */ }
  }

  /* ------------------------------------------------------------ 主流程 */

  let inFlight = null;

  async function load(force, useGps) {
    if (inFlight) return inFlight;

    if (!force) {
      const cached = readCache();
      if (cached) {
        render(cached.data, cached.city);
        return;
      }
    }

    root.dataset.state = 'loading';
    loading.hidden = false;
    content.hidden = true;
    errorBox.hidden = true;
    loadingText.textContent = t('locating');

    inFlight = (async () => {
      let location = null;

      if (useGps) location = await locateByBrowser();
      if (!location) location = await locateByIp();
      if (!location) {
        location = { lat: defaultLat, lon: defaultLon, city: defaultCity, source: 'default' };
      }

      const city = location.city || (location.source === 'default' ? defaultCity : null);

      try {
        const data = await fetchWeather(location.lat, location.lon);
        render(data, city);
        writeCache({ data, city });
      } catch (err) {
        console.error('weather: failed to load', err);
        showError('error');
      }
    })().finally(() => {
      inFlight = null;
    });

    return inFlight;
  }

  /* ------------------------------------------------------------ 交互 */

  function openPanel() {
    if (!panel) return;
    panel.hidden = false;
    // 强制回流后再添加 class，保证过渡动画生效
    void panel.offsetWidth;
    panel.classList.add('is-open');
    toggleBtn.setAttribute('aria-expanded', 'true');
    root.classList.add('is-open');
  }

  function closePanel() {
    if (!panel) return;
    panel.classList.remove('is-open');
    toggleBtn.setAttribute('aria-expanded', 'false');
    root.classList.remove('is-open');
    // 等待过渡结束后隐藏，避免动画被截断
    setTimeout(() => {
      if (!panel.classList.contains('is-open')) panel.hidden = true;
    }, 260);
  }

  toggleBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (root.classList.contains('is-open')) closePanel();
    else openPanel();
  });

  panel?.addEventListener('click', (e) => e.stopPropagation());

  document.addEventListener('click', (e) => {
    if (!root.classList.contains('is-open')) return;
    if (root.contains(e.target)) return;
    closePanel();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && root.classList.contains('is-open')) closePanel();
  });

  refreshBtn?.addEventListener('click', () => {
    refreshBtn.classList.add('is-spinning');
    setTimeout(() => refreshBtn.classList.remove('is-spinning'), 700);
    try { sessionStorage.removeItem(CACHE_KEY); } catch (_) { /* ignore */ }
    load(true, false);
  });

  locateBtn?.addEventListener('click', () => {
    try { sessionStorage.removeItem(CACHE_KEY); } catch (_) { /* ignore */ }
    load(true, true);
  });

  /* 语言切换：默认英文，中文可选 */
  root.querySelectorAll('.weather-plugin__lang-btn').forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.lang === lang);
    btn.setAttribute('aria-pressed', String(btn.dataset.lang === lang));

    btn.addEventListener('click', () => {
      const next = btn.dataset.lang;
      if (next === lang) return;
      lang = next;

      try { localStorage.setItem(LANG_KEY, lang); } catch (_) { /* ignore */ }

      root.querySelectorAll('.weather-plugin__lang-btn').forEach((b) => {
        const active = b.dataset.lang === lang;
        b.classList.toggle('is-active', active);
        b.setAttribute('aria-pressed', String(active));
      });

      applyStaticText();
      // 重新渲染以更新天气描述、风向与时间格式
      const cached = readCache();
      if (cached) render(cached.data, cached.city);
    });
  });

  /* 时钟 */
  function tickClock() {
    const now = new Date();
    const time = el('weather-plugin-time');
    const date = el('weather-plugin-date');
    if (time) {
      time.textContent = now.toLocaleTimeString(lang === 'zh' ? 'zh-CN' : 'en-US', {
        hour: '2-digit', minute: '2-digit', hour12: lang === 'en'
      });
    }
    if (date) {
      date.textContent = now.toLocaleDateString(lang === 'zh' ? 'zh-CN' : 'en-US', {
        weekday: 'short', month: 'short', day: 'numeric'
      });
    }
  }

  buildParticles();
  tickClock();
  setInterval(tickClock, 30000);
  applyStaticText();

  load(false, false);
})();
