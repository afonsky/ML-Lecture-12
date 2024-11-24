---
theme: seriph
addons:
  - "@twitwi/slidev-addon-ultracharger"
addonsConfig:
  ultracharger:
    inlineSvg:
      markersWorkaround: false
    disable:
      - metaFooter
      - tocFooter
NObackground: >-
  https://images.unsplash.com/photo-1511149755252-35875b273fd6?ixlib=rb-4.0.3&dl=leon-contreras-qpdfU6vehgs-unsplash.jpg&w=1920&q=80&fm=jpg&crop=entropy&cs=tinysrgb
background: /logo/ship2.jpg
highlighter: shiki
routerMode: hash
lineNumbers: false

css: unocss
title: Machine Learning
subtitle: High-Dimensional Problems
date: 25/11/2024
venue: HSE
author: Alexey Boldyrev
---

# <span style="font-size:28.0pt" v-html="$slidev.configs.title?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:32.0pt" v-html="$slidev.configs.subtitle?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:18.0pt" v-html="$slidev.configs.author?.replaceAll(' ', '<br/>')"></span>

<span style="font-size:18.0pt" v-html="$slidev.configs.date?.replaceAll(' ', '<br/>')"></span>

<div>
<br>
<span style="color:#b3b3b3ff; font-size: 11px; float: right;">Image credit: ‘The Mayﬂower at Sea’<br> by Granville Perkins, 1876<br>
Wallach Division Picture Collection<br> The New York Public Library.
</span>
</div>

<style>
  :deep(footer) { padding-bottom: 3em !important; }
</style>

<!--
NB: This demo uses a custom syntax (using preparser extensions), with all the @@@@.
-->

---
src: ./slides/0_attendance.md
---

---
src: ./slides/1_example_DNA.md
---

---
src: ./slides/2_examples.md
---

---
src: ./slides/6_SVM.md
---

---
src: ./slides/0_end.md
---
