---
title: 维基
icon: fas fa-book
order: 4
permalink: /wiki/
---

> 记多少命令和快捷键会让脑袋爆炸呢？

<ul>
{% assign wlist = site.wiki | sort: 'title' %}
{% for w in wlist %}
<li><a href="{{ w.url | relative_url }}">{{ w.title }}</a></li>
{% endfor %}
</ul>
