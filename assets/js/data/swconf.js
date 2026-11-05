---
layout: compress
permalink: '/:path/swconf.js'
# Note that this file will be fetched by the ServiceWorker, so it will not be cached.
---

const swconf = {
  {% if site.pwa.cache.enabled %}
    cacheName: 'chirpy-{{ "now" | date: "%s" }}',

    {%- comment -%} Resources added to the cache during PWA installation. {%- endcomment -%}
    resources: [
      '{{ "/assets/css/:THEME.css" | replace: ':THEME', site.theme | relative_url }}',
      '{{ "/" | relative_url }}',
      {% for tab in site.tabs %}
        '{{- tab.url | relative_url -}}',
      {% endfor %}

      {%- comment -%} webmeji character (bottom-right mascot on every page). {%- endcomment -%}
      '{{ "/assets/webmeji/webmeji.css" | relative_url }}',
      '{{ "/assets/webmeji/config.js" | relative_url }}',
      '{{ "/assets/webmeji/webmeji.js" | relative_url }}',
      {%- comment -%} sprite frames, so the character works offline. {%- endcomment -%}
      {%- for _frame in site.static_files -%}
        {%- if _frame.path contains "/assets/webmeji/miku/" and _frame.extname == ".png" -%}
          '{{ _frame.path | relative_url }}',
        {%- endif -%}
      {%- endfor -%}

      {%- comment -%} weather widget (top-right glass panel) + Meteocons icons. {%- endcomment -%}
      '{{ "/assets/css/weather.css" | relative_url }}',
      '{{ "/assets/js/weather.js" | relative_url }}',
      {%- comment -%} Meteocons animated SVG icons, so the widget works offline. {%- endcomment -%}
      {%- for _icon in site.static_files -%}
        {%- if _icon.path contains "/assets/img/weather/" and _icon.extname == ".svg" -%}
          '{{ _icon.path | relative_url }}',
        {%- endif -%}
      {%- endfor -%}

      {% assign cache_list = site.static_files | where: 'swcache', true %}
      {% for file in cache_list %}
        '{{ file.path | relative_url }}'{%- unless forloop.last -%},{%- endunless -%}
      {% endfor %}
    ],

    interceptor: {
      {%- comment -%} URLs containing the following paths will not be cached. {%- endcomment -%}
      paths: [
        {% for path in site.pwa.cache.deny_paths %}
          {% unless path == empty %}
            '{{ path | relative_url }}'{%- unless forloop.last -%},{%- endunless -%}
          {% endunless %}
        {% endfor %}
      ],

      {%- comment -%} URLs containing the following prefixes will not be cached. {%- endcomment -%}
      urlPrefixes: [
        {% if site.analytics.goatcounter.id != nil and site.pageviews.provider == 'goatcounter' %}
          'https://{{ site.analytics.goatcounter.id }}.goatcounter.com/counter/'
        {% endif %}
      ]
    },

    purge: false
  {% else %}
    purge: true
  {% endif %}
};
