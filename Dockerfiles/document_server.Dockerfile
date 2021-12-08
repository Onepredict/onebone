FROM nginx:stable

COPY ./tools/nginx/default.conf /etc/nginx/conf.d
CMD ["nginx", "-g", "daemon off;"]
