---
layout: categories
icon: fas fa-stream
order: 1
---
{% for post in sortedPosts %}
  <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}