# 已测结果表（自动生成）

Quality frames 0..299 macro mean; decimal MB; timing 1..299 pooled; P95 nearest-rank. N3DV Flash best-of-two, others both; MeetRoom only discussion/trimming, Flash/CL one run, others two. No matched hardware/platform claim across servers.

## N3DV

|方法|PSNR ↑|SSIM ↑|LPIPS ↓|完整 MB ↓|FPS ↑|平均 ms ↓|P95 ms ↓|GPU Kernel ms ↓|测速口径|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|QUEEN|32.3012|0.948921|0.093566|5906.70|0.628|1591.477|2244.603|—|two_runs_pooled|
|3DGStream|31.1118|0.941987|0.100224|2447.07|35.407|28.243|77.456|—|two_runs_pooled|
|4DGC-CL|31.0705|0.942525|0.095288|260.69|1.069|935.750|975.474|—|two_runs_pooled|
|Flash4DGC|31.0705|0.942525|0.095288|355.06|33.828|29.561|21.764|0.10085|best_of_two_per_scene|

### 逐场景

|方法 / 场景|PSNR ↑|SSIM ↑|LPIPS ↓|完整 MB ↓|FPS ↑|平均 ms ↓|P95 ms ↓|GPU Kernel ms ↓|测速口径|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|QUEEN / coffee_martini|28.4522|0.918545|0.126188|8119.65|0.443|2256.004|2429.360|—|two_runs_pooled|
|QUEEN / cook_spinach|33.6406|0.958170|0.084706|5188.72|0.719|1390.098|1509.479|—|two_runs_pooled|
|QUEEN / cut_roasted_beef|33.9519|0.960803|0.084107|5214.02|0.713|1402.367|1483.014|—|two_runs_pooled|
|QUEEN / flame_salmon|29.2571|0.926648|0.113691|7834.38|0.468|2135.743|2266.869|—|two_runs_pooled|
|QUEEN / flame_steak|34.3622|0.964355|0.077022|4581.75|0.828|1207.176|1315.672|—|two_runs_pooled|
|QUEEN / sear_steak|34.1434|0.965007|0.075685|4501.68|0.864|1157.475|1244.277|—|two_runs_pooled|
|3DGStream / coffee_martini|28.1576|0.914997|0.119729|2449.38|27.192|36.775|82.673|—|two_runs_pooled|
|3DGStream / cook_spinach|31.4152|0.947473|0.105592|2457.11|54.153|18.466|12.338|—|two_runs_pooled|
|3DGStream / cut_roasted_beef|32.9628|0.953577|0.088249|2419.21|22.985|43.506|83.935|—|two_runs_pooled|
|3DGStream / flame_salmon|27.5078|0.918681|0.111743|2443.20|29.178|34.273|87.882|—|two_runs_pooled|
|3DGStream / flame_steak|33.9001|0.959439|0.083173|2468.59|56.291|17.765|13.239|—|two_runs_pooled|
|3DGStream / sear_steak|32.7273|0.957753|0.092860|2444.94|53.553|18.673|12.059|—|two_runs_pooled|
|4DGC-CL / coffee_martini|28.0320|0.914638|0.121450|278.62|1.065|938.886|952.749|—|two_runs_pooled|
|4DGC-CL / cook_spinach|32.6581|0.951221|0.090990|250.06|1.049|953.061|1009.129|—|two_runs_pooled|
|4DGC-CL / cut_roasted_beef|33.1680|0.954573|0.086958|260.25|1.087|920.239|939.754|—|two_runs_pooled|
|4DGC-CL / flame_salmon|27.5392|0.918156|0.114400|266.15|1.074|931.176|969.755|—|two_runs_pooled|
|4DGC-CL / flame_steak|33.0234|0.959634|0.076987|249.27|1.069|935.736|980.303|—|two_runs_pooled|
|4DGC-CL / sear_steak|32.0020|0.956931|0.080945|259.79|1.069|935.405|977.471|—|two_runs_pooled|
|Flash4DGC / coffee_martini|28.0320|0.914638|0.121450|369.76|32.243|31.014|21.892|0.10340|best_of_two_per_scene|
|Flash4DGC / cook_spinach|32.6581|0.951221|0.090990|347.67|34.765|28.764|21.617|0.09858|best_of_two_per_scene|
|Flash4DGC / cut_roasted_beef|33.1680|0.954573|0.086958|352.21|35.202|28.408|17.850|0.10298|best_of_two_per_scene|
|Flash4DGC / flame_salmon|27.5392|0.918156|0.114400|361.95|31.246|32.004|23.782|0.09848|best_of_two_per_scene|
|Flash4DGC / flame_steak|33.0234|0.959634|0.076987|344.57|34.570|28.927|23.448|0.10175|best_of_two_per_scene|
|Flash4DGC / sear_steak|32.0020|0.956931|0.080945|354.21|35.400|28.248|18.536|0.09993|best_of_two_per_scene|

## MeetRoom-2

|方法|PSNR ↑|SSIM ↑|LPIPS ↓|完整 MB ↓|FPS ↑|平均 ms ↓|P95 ms ↓|GPU Kernel ms ↓|测速口径|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|QUEEN|25.9243|0.906654|0.173001|3612.41|0.939|1065.222|1313.798|—|two_runs_pooled|
|3DGStream|27.4743|0.911195|0.145024|2424.56|25.415|39.346|63.098|—|two_runs_pooled|
|4DGC-CL|27.5596|0.913102|0.142732|282.96|1.167|856.903|878.574|—|single_run|
|Flash4DGC|27.5596|0.913102|0.142732|374.26|36.667|27.272|17.000|0.10125|single_run|

### 逐场景

|方法 / 场景|PSNR ↑|SSIM ↑|LPIPS ↓|完整 MB ↓|FPS ↑|平均 ms ↓|P95 ms ↓|GPU Kernel ms ↓|测速口径|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|QUEEN / discussion|26.7073|0.908844|0.168914|4204.39|0.823|1215.341|1353.364|—|two_runs_pooled|
|QUEEN / trimming|25.1414|0.904464|0.177088|3020.43|1.093|915.104|1258.754|—|two_runs_pooled|
|3DGStream / discussion|27.2786|0.909590|0.148604|2432.21|24.627|40.606|65.332|—|two_runs_pooled|
|3DGStream / trimming|27.6701|0.912800|0.141443|2416.90|26.256|38.086|59.698|—|two_runs_pooled|
|4DGC-CL / discussion|27.3521|0.911720|0.146537|289.87|1.158|863.296|880.653|—|single_run|
|4DGC-CL / trimming|27.7672|0.914483|0.138927|276.05|1.176|850.510|874.309|—|single_run|
|Flash4DGC / discussion|27.3521|0.911720|0.146537|383.11|35.302|28.327|17.304|0.10134|single_run|
|Flash4DGC / trimming|27.7672|0.914483|0.138927|365.41|38.142|26.218|16.455|0.10115|single_run|

## N3DV 统一两轮合并口径（敏感性对照）

|方法|PSNR ↑|SSIM ↑|LPIPS ↓|完整 MB ↓|FPS ↑|平均 ms ↓|P95 ms ↓|GPU Kernel ms ↓|测速口径|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|QUEEN|32.3012|0.948921|0.093566|5906.70|0.628|1591.477|2244.603|—|two_runs_pooled|
|3DGStream|31.1118|0.941987|0.100224|2447.07|35.407|28.243|77.456|—|two_runs_pooled|
|4DGC-CL|31.0705|0.942525|0.095288|260.69|1.069|935.750|975.474|—|two_runs_pooled|
|Flash4DGC|31.0705|0.942525|0.095288|355.06|26.395|37.886|56.579|0.10085|two_runs_pooled|
