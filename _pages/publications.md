---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}


{% capture collapsed_content %}
Your collapsed content goes here.
{% endcapture %}

<div class="collapsed-element">
  {{ collapsed_content | markdownify }}
</div>

<script>
  // JavaScript to toggle the "expanded" class on click
  document.querySelector('.collapsed-element').addEventListener('click', function() {
    this.classList.toggle('expanded');
  });
</script>
