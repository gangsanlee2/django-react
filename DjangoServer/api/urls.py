"""api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from api.views import hello

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('',hello),
    path("api/analysis/", include('api.analysis.urls')),
    path("api/dlearn/", include('api.dlearn.urls')),
    path("api/mlearn/", include('api.mlearn.urls')),
    path("api/nlp/", include('api.nlp.urls')),
    path("api/vision/", include('api.vision.urls')),
    path("blog/comments/", include('blog.comments.urls')),
    path("blog/posts/", include('blog.posts.urls')),
    path("blog/tags/", include('blog.tags.urls')),
    path("blog/views/", include('blog.views.urls')),
    path("common/main/", include('blog.views.urls')),
    path("multiplex/cinemas/", include('multiplex.cinemas.urls')),
    path("multiplex/movies/", include('multiplex.movies.urls')),
    path("multiplex/showtimes/", include('multiplex.showtimes.urls')),
    path("multiplex/theater_tickets/", include('multiplex.theater_tickets.urls')),
    path("multiplex/theaters/", include('multiplex.theaters.urls')),
    path("security/users/", include('security.users.urls')),
    path("shop/carts/", include('multiplex.movies.urls')),
    path("shop/categories/", include('shop.categories.urls')),
    path("shop/deliveries/", include('shop.deliveries.urls')),
    path("shop/orders/", include('shop.orders.urls')),
    path("shop/products/", include('shop.products.urls')),
]
