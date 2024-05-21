import Vue from 'vue'
import App from './App.vue'
// 导入iview的js文件
import iView from 'iview'
// 导入iview的css文件
import 'iview/dist/styles/iview.css'

Vue.use(iView)


Vue.config.productionTip = false

new Vue({
  render: h => h(App),
}).$mount('#app')
