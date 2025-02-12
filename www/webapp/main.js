"use strict"

//import ImageHolder from "./image_holder.js"

let deconv_type = "clean_modified"
let deconv_name = "test_deconvolver"
let deconv_complete = false

let scratch_canvas = document.getElementById("canvas")
let scratch_canvas_ctx = scratch_canvas.getContext("2d")
scratch_canvas_ctx.imageSmoothingEnabled = false // NOTE: this needs to be set after each canvas resize

let sci_file_picker = document.getElementById("sci_file")
let psf_file_picker = document.getElementById("psf_file")

let sci_canvas = document.getElementById("sci_canvas")
let psf_canvas = document.getElementById("psf_canvas")

let adjusted_psf_canvas = document.getElementById("adjusted-psf-canvas")

let clean_map_canvas = document.getElementById("clean_map_canvas")
let residual_canvas = document.getElementById("residual_canvas")

let download_clean_map_button = document.getElementById("download-clean-map-button")
let download_residual_button = document.getElementById("download-residual-button")

let run_deconv_button = document.getElementById("run_deconv")
let n_max_iter_field = document.getElementById("n_max_iter")


let deconv_status_mgr = new StatusKVManager([
	["Science Observation Uploaded", false, {"is-good":false}, {highlight_on_hover_target:document.getElementById("science-target-button")}],
	["PSF Observation Uploaded", false, {"is-good":false}, {highlight_on_hover_target:document.getElementById("psf-target-button")}],
	["Parameters Validated", false, {"is-good":false}, {highlight_on_hover_target:document.getElementById("param-container")}],
	["Deconvolution Running", false, {"is-good":undefined}, {highlight_on_hover_target:document.getElementById("run_deconv")}],
	["Current Deconvolution Image Layer", "No Previous Run", {"is-good":undefined}],
	["Deconvolution Iteration", "No Previous Run", {"is-good":undefined}],
	["Last Plot Update Iteration", "No Previous Run", {"is-good":undefined}, {highlight_on_hover_target:document.getElementById("progress-plot-region")}],
	["Results Available", false, {"is-good":false}, {highlight_on_hover_target:document.getElementById("results-region")}]
])
deconv_status_mgr.addTo(document.getElementById("status-container"))


let sci_image_holder = new ImageHolder(
	sci_canvas, 
	sci_file_picker, 
	"change",
	()=>{
		deconv_status_mgr.set("Science Observation Uploaded", true, {"is-good":true})
	}
)
let psf_image_holder = new ImageHolder(
	psf_canvas, 
	psf_file_picker, 
	"change",
	()=>{
		deconv_status_mgr.set("PSF Observation Uploaded", true, {"is-good":true})
	}
)
let deconv_clean_map = null
let deconv_residual = null

let clean_modified_params = new CleanModifiedParameters(document.getElementById("param-container"))


function setHideViaDetails(detail_element_name, container_element_name, show_at_start_flag=false){
	let details = document.getElementById(detail_element_name)
	let container = document.getElementById(container_element_name)
	
	container.hide = ()=>{
		container.prev_style = container.getAttribute("style")
		if (container.prev_style === null){
			container.prev_style = "height:auto;overflow:visible"
		}
		container.setAttribute("style","height:0px;overflow:hidden")
	}
	container.show = ()=>{
		container.setAttribute("style", container.prev_style)
	}
	
	details.onclick = (e)=>{
		if(details.hasAttribute("open")){
			// We are closing the details
			container.hide()
		}
		else {
			container.show()
		}
	}
	
	container.hide()
	
	if(show_at_start_flag){
		details.dispatchEvent(new Event("click"))
		details.setAttribute("open", null)
	}
}

setHideViaDetails("results-details", "results")
setHideViaDetails("progress-plot-details", "plot-container-1")

let dataset_cpp_array_cache = new Map();
let plot_name_map = new Map();

let fig_shape = V.from(48,32)
let fig_units = "rem"

let figure = new Figure({
	container : document.getElementById("progress-plots"),
	shape : fig_shape,
	units : fig_units
})

figure.appendPlotAreaFromRect("stopping_criteria", new R(0.05,0.05,0.9,0.9))
figure.plot_areas.get("stopping_criteria").appendDataAreaFromRect("stopping_criteria_data_area", new R(0.1,0.1,0.8,0.8))
figure.plot_areas.get("stopping_criteria").appendAxesFromExtent("stopping_criteria_axes", E.from(NaN,NaN,NaN,NaN), {axis_names : ["iteration", "fabs (red) and rms (green) value"]})
figure.plot_areas.get("stopping_criteria").appendAxesFromExtent("stopping_criteria_axes_2", E.from(NaN,NaN,NaN,NaN), {axis_positions : [null, 1], axis_names : ["iteration", "threshold (purple) value"]})

//figure.plot_areas.get("fabs_record").appendAxesFromExtent("fabs_axes", E.from(-1.2,3.7,-0.9,2))
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes", "fabs_record")
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes", "rms_record")
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes_2", "threshold_record")
figure.set_title("Stopping Criteria and Threshold for each Iteration")
figure.set_caption([
"Figure 5: Left axis shows two stopping criteria: 1) <em>fabs</em> (red), the absolute brightest pixel of the residual \
divided by the absolute brightest pixel of the original image; 2) <em>rms</em> (green), the root-mean-square of the residual \
divided by the root-mean-square of the original image. Right axis shows the threshold (purple) which is used to choose the \
<em>selected pixels</em> (Fig. 2).",

"The two stopping criteria <em>fabs</em> and <em>rms</em> progress from a value of 1 to whatever value is specified in \
<em>fabs_frac_threshold</em> and <em>rms_frac_threshold</em> respectively. When they reach their respective thresholds, \
iteration is terminated. Ideally the graphs of the stopping criteria should be monotonically decreasing, having upwards \
movement of these lines implies that the algorithm is struggling to find a solution.",

"The threshold for each iteration may jump around if <em>adaptive_threshold_flag</em> is set, as different parts of the \
residual are classified as 'background' and 'foreground'. If <em>adpative_threshold_flag</em> is not set, the graph should \
monotonically decrease as the <em>threshold</em> is a constant fraction of the absolute brightest pixel of the residual."

])


let figure2 = new Figure({
	container : document.getElementById("progress-plots"),
	shape : fig_shape,
	units : fig_units
})
figure2.appendPlotAreaFromRect("residual_histogram",  new R(0.05,0.05,0.9,0.9))
figure2.plot_areas.get("residual_histogram").appendDataAreaFromRect("residual_histogram_data_area", new R(0.1,0.1,0.8,0.8))
figure2.plot_areas.get("residual_histogram").appendAxesFromExtent("residual_histogram_axes", E.from(NaN,NaN,NaN,NaN), {axis_names : ["value (vertical line is the current step's threshold value)", "count"], nonlinear_transform : log_transform_y})
figure2.plot_areas.get("residual_histogram").newDatasetForAxes("residual_histogram_axes", "residual_histogram_data")
figure2.plot_areas.get("residual_histogram").setDatasetPlotTypeArtist("residual_histogram_data", new StepPlotArtist())
figure2.plot_areas.get("residual_histogram").addDatasetToAxes("residual_histogram_axes", new RingbufferDataset("threshold_line_data",1))
figure2.plot_areas.get("residual_histogram").setDatasetPlotTypeArtist("threshold_line_data", new VlinePlotArtist())
figure2.set_title("Histogram of Residual at current Iteration")
figure2.set_caption([
"Figure 6: Step graph shows the histogram of the <em>residual</em> at the current iteration. The vertical line shows the <em>threshold</em> \
of the current iteration. The histogram should initially be the same as the histogram of the original image, typically this is a high count \
of low value pixels (the field) and a similar (but often lesser) count of high value pixels (the object). The boundary between the \
two populations may be flattened due to blurring of the object. In later iterations, as the object is removed from the residual, the graph should \
resemble the graph of a gaussian (in log space) centered on zero with a variance depending on the noise in the image."
])


let figure3 = new Figure({
	container : document.getElementById("progress-plots"),
	shape : fig_shape,
	units : fig_units
})
figure3.appendPlotAreaFromRect("component_histogram",  new R(0.05,0.05,0.9,0.9))
figure3.plot_areas.get("component_histogram").appendDataAreaFromRect("component_data_area", new R(0.1,0.1,0.8,0.8))
figure3.plot_areas.get("component_histogram").appendAxesFromExtent("component_axes", E.from(NaN,NaN,NaN,NaN), {axis_names : ["value", "count"], nonlinear_transform : log_transform_y})
figure3.plot_areas.get("component_histogram").newDatasetForAxes("component_axes", "component_data")
figure3.plot_areas.get("component_histogram").setDatasetPlotTypeArtist("component_data", new StepPlotArtist())
figure3.set_title("Histogram of Components at current Iteration")
figure3.set_caption([
"Figure 7: Step graph shows the histogram of the <em>current components</em> (Fig. 4). Initially this will all be zeros, and typically evolve into \
a bi-modal distribution of dark 'background' pixels, and bright 'foreground' pixels."
])

let figure4 = new Figure({
	container : document.getElementById("progress-plots"),
	shape : fig_shape,
	units : fig_units
})
figure4.appendPlotAreaFromRect("selected_pixels_histogram", new R(0.05,0.05,0.9,0.9))
figure4.plot_areas.get("selected_pixels_histogram").appendDataAreaFromRect("sp_data_area", new R(0.1,0.1,0.8,0.8))
figure4.plot_areas.get("selected_pixels_histogram").appendAxesFromExtent("sp_data_area", E.from(NaN,NaN,NaN,NaN), {axis_names : ["value", "count"], nonlinear_transform : log_transform_y})
figure4.plot_areas.get("selected_pixels_histogram").newDatasetForAxes("sp_data_area", "selected_pixels_data")
figure4.plot_areas.get("selected_pixels_histogram").setDatasetPlotTypeArtist("selected_pixels_data", new StepPlotArtist())
figure4.set_title("Histogram of selected pixels at current Iteration")
figure4.set_caption([
"Figure 8: Step graph shows the histogram of the <em>selected pixels</em> for the current iteration (Fig. 2). Should always consist of the section \
of the <em>residual</em> histogram (Fig. 6) above the current <em>threshold</em>."
])

let figure5 = new Figure({
	container : document.getElementById("progress-plots"),
	shape : fig_shape,
	units : fig_units
})
figure5.appendPlotAreaFromRect("current_convolved_histogram", new R(0.05,0.05,0.9,0.9))
figure5.plot_areas.get("current_convolved_histogram").appendDataAreaFromRect("cc_data_area", new R(0.1,0.1,0.8,0.8))
figure5.plot_areas.get("current_convolved_histogram").appendAxesFromExtent("cc_data_area", E.from(NaN,NaN,NaN,NaN), {axis_names : ["value", "count"], nonlinear_transform : log_transform_y})
figure5.plot_areas.get("current_convolved_histogram").newDatasetForAxes("cc_data_area", "current_convolved_data")
figure5.plot_areas.get("current_convolved_histogram").setDatasetPlotTypeArtist("current_convolved_data", new StepPlotArtist())
figure5.set_title("Histogram of the 'current convolved', i.e., the selected pixels of the current Iteration convolved with the PSF")
figure5.set_caption([
"Figure 9: Step graph shows the histogram of the <em>current convolved</em> (Fig. 3), i.e., the <em>currently</em> selected pixels <em>convolved</em> \
with the PSF. This should look very similar to the <em>selected pixels</em> histogram (Fig. 8), but with smoothed edges."
]);


plot_name_map.set("stopping_criteria", figure.plot_areas.get("stopping_criteria"))
plot_name_map.set("residual_histogram", figure2.plot_areas.get("residual_histogram"))
plot_name_map.set("component_histogram", figure3.plot_areas.get("component_histogram"))
plot_name_map.set("selected_pixels_histogram", figure4.plot_areas.get("selected_pixels_histogram"))
plot_name_map.set("current_convolved_histogram", figure5.plot_areas.get("current_convolved_histogram"))


download_clean_map_button.addEventListener(
	"click",
	new WasmDataDownloader( 
		(file_id)=>{
			let fn = ""
			let ext = ".tiff"
			if(sci_image_holder.file.name.endsWith(".tiff")){
				fn = sci_image_holder.file.name.slice(0,-5)+"_"
			}
			else if (sci_image_holder.file.name.endsWith(".tif")){
				fn = sci_image_holder.file.name.slice(0,-4)+"_"
				ext = ".tif"
			}
			
			console.log(deconv_type, deconv_name, file_id)
			const data_min = Module.get_data_min(deconv_type, deconv_name, file_id)
			const data_max = Module.get_data_max(deconv_type, deconv_name, file_id)
			
			return fn+`deconvolved_clean_map__min_${data_min}_max_${data_max}` + ext
		},
		"deconv.clean_map", 
		(file_id)=>Module.get_tiff(deconv_type, deconv_name, file_id, sci_image_holder.name),
		()=>{if(!deconv_complete){alert("Deconvolution not completed, cannot download results");} return deconv_complete;},
	)
)

download_residual_button.addEventListener(
	"click",
	new WasmDataDownloader( 
		(file_id)=>{
			let fn = ""
			let ext=".tiff"
			if(sci_image_holder.file.name.endsWith(".tiff")){
				fn = sci_image_holder.file.name.slice(0,-5)+"_"
			}
			else if (sci_image_holder.file.name.endsWith(".tif")){
				fn = sci_image_holder.file.name.slice(0,-4)+"_"
				ext = ".tif"
			}
			
			const data_min = Module.get_data_min(deconv_type, deconv_name, file_id)
			const data_max = Module.get_data_max(deconv_type, deconv_name, file_id)
			
			return fn+`deconvolved_residual__min_${data_min}_max_${data_max}`+ext
		},
		"deconv.residual", 
		(file_id)=>Module.get_tiff(deconv_type, deconv_name, file_id, sci_image_holder.name),
		()=>{if(!deconv_complete){alert("Deconvolution not completed, cannot download results");} return deconv_complete;},
	)
)

run_deconv_button.addEventListener("click", 
	async (e)=>{
		try{
			e.target.textContent = "Deconvolution in progress..."
			e.target.disabled = true
			if ((sci_image_holder.name === null) || (psf_image_holder.name === null)) {
				alert("ERROR: Missing input data.\n\nDeconvolution requires upload of a science image and a psf image")
				return
			}

			let invalid_params = clean_modified_params.validate()
			if(invalid_params.length != 0){
				alert(`ERROR: Could not run deconvolution.\n\nThe following parameters are invalid and need to be corrected:\n\t${invalid_params.join("\n\t")}`)
				return;
			}
			
			deconv_complete = false
			deconv_status_mgr.set("Deconvolution Running", true)
			deconv_status_mgr.set("Results Available", false, {"is-good":false})
			console.log("Creating deconvolver")
			
			await Module.create_deconvolver(deconv_type, deconv_name)

			// Check for invalid params again when we set the values
			invalid_params = clean_modified_params.set_params(deconv_type, deconv_name)
			if(invalid_params.length != 0){
				alert(`ERROR: Could not run deconvolution.\n\nThe following parameters are invalid and need to be corrected:\n\t${invalid_params.join("\n\t")}`)
				return;
			}
			
			//console.log("run_deconv_button.addEventListener::click", Math.log10(clean_modified_params.valueOf("fabs_frac_threshold")))

			console.log(`Preparing deconvolver for ${sci_image_holder.name} ${psf_image_holder.name}`)
			let err_msg = await Module.prepare_deconvolver(deconv_type, deconv_name, sci_image_holder.name, psf_image_holder.name, "")
			
			if (err_msg.length >0){
				alert(`ERROR: ${err_msg}`)
				return
			}
			
			
			// Clear plots
			for(const plot_area of plot_name_map.values()){
				plot_area.clear()
			}
			
			
			
			console.log("Running prepared deconvolver")
			await Module.run_deconvolver(deconv_type, deconv_name)
			
			deconv_complete = true
			deconv_status_mgr.set("Deconvolution Running", false)
			
			let width = sci_image_holder.im_w
			let height = sci_image_holder.im_h
			
			// Updated webassembly so results are the same shape as
			// input science image. Therefore do not need to change to
			// odd-shape.
			//width += (1-width%2)
			//height += (1-height%2)

			console.log("Get results from deconvolver")
			// assume results are the same size as the science image
			deconv_clean_map = getImageDataFromResult(
				Module.get_deconvolver_clean_map, 
				[deconv_type, deconv_name], 
				width, 
				height
			)
			deconv_residual = getImageDataFromResult(
				Module.get_deconvolver_residual,
				[deconv_type, deconv_name],
				width,
				height
			)

			console.log("deconv_clean_map", deconv_clean_map)
			console.log("deconv_residual", deconv_residual)

			console.log("Display results on canvas elements")

			clean_map_canvas.width = width
			clean_map_canvas.height = height
			clean_map_canvas.getContext("2d").putImageData(deconv_clean_map,0,0)

		
			residual_canvas.width = width
			residual_canvas.height = height
			residual_canvas.getContext("2d").putImageData(deconv_residual,0,0)	

			console.log("Results should be displayed")
			deconv_status_mgr.set("Results Available", true, {"is-good":true})
		}
		finally {
			e.target.textContent = "Run Deconvolution"
			e.target.disabled = false
		}
	}
)





