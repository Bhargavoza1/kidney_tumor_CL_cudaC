// src/setupProxy.js

const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  //console.log("urld : " , process.env.REACT_APP_DEV);
  if (process.env.REACT_APP_BACKEND_URL){
    app.use(
      '/api',
      createProxyMiddleware({
        target: process.env.REACT_APP_BACKEND_URL, // Backend URL
        changeOrigin: true,
      })
    );}
};
