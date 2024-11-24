import { defineAppSetup } from '@slidev/types'

/* plotly */
import { VuePlotly } from 'vue3-plotly'

/* VueTableLite */
// import VueTableLite from 'vue3-table-lite/ts'

// import { VueImageZoomer } from 'vue-image-zoomer'
// import 'vue-image-zoomer/dist/style.css'

// import { InnerImageZoom } from 'vue-inner-image-zoom'

// import { ZoomOnHover } from "vue-zoom-on-hover"

export default defineAppSetup(({ app, router }) => {
  app.component('v-plotly', VuePlotly  )
  // app.component('vue-image-zoomer', VueImageZoomer  )
  // app.component('inner-image-zoom', InnerImageZoom  )
  // app.component('zoom-on-hover', ZoomOnHover  )
  // app.component('table-lite', VueTableLite  )
})