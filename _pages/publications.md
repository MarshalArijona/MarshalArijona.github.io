---
layout: page
permalink: /publications/
title: Publications
description: publications by chronological order.
years: [2020, 2021, 2022, 2023, 2024, 2025]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
