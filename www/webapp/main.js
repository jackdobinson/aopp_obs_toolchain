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
	["Science Observation Uploaded", false, {"is-good":false}],
	["PSF Observation Uploaded", false, {"is-good":false}],
	["Parameters Validated", false, {"is-good":false}],
	["Deconvolution Running", false, {"is-good":undefined}],
	["Deconvolution Iteration", "No Previous Run", {"is-good":undefined}],
	["Last Plot Update Iteration", "No Previous Run", {"is-good":undefined}],
	["Results Available", false, {"is-good":false}]
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


let progress_plot_details = document.getElementById("progress-plot-details")
let plot_container_1 = document.getElementById("plot-container-1")
plot_container_1.hide = ()=>{
	plot_container_1.setAttribute("style","height:0px;overflow:hidden")
}
plot_container_1.show = ()=>{
	plot_container_1.setAttribute("style","height:auto;overflow:visible")
}
plot_container_1.hide()

progress_plot_details.onclick = (e)=>{
	
	if(progress_plot_details.hasAttribute("open")){
		// We are closing the details
		plot_container_1.hide()
	}
	else {
		plot_container_1.show()
	}
}
/*
let svg_figure = new SvgFig(
	"progress plots", 
	document.getElementById("progress-plots"),
	V.from(20,20),
	new R(0,0,2,2)
)
*/
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
figure.plot_areas.get("stopping_criteria").appendAxesFromExtent("stopping_criteria_axes", E.from(NaN,NaN,NaN,NaN), {axis_names : ["iteration", "fabs/rms value"]})
figure.plot_areas.get("stopping_criteria").appendAxesFromExtent("stopping_criteria_axes_2", E.from(NaN,NaN,NaN,NaN), {axis_positions : [null, 1], axis_names : ["iteration", "threshold value"]})

//figure.plot_areas.get("fabs_record").appendAxesFromExtent("fabs_axes", E.from(-1.2,3.7,-0.9,2))
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes", "fabs_record")
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes", "rms_record")
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes_2", "threshold_record")

figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes", "test_1_dataset")
figure.plot_areas.get("stopping_criteria").newDatasetForAxes("stopping_criteria_axes_2", "test_2_dataset")

figure.plot_areas.get("stopping_criteria").setDatasetPlotTypeArtist("test_2_dataset", new StepPlotArtist())

figure.plot_areas.get("stopping_criteria").addDatasetToAxes("stopping_criteria_axes_2", new RingbufferDataset("test_3_dataset",1))
figure.plot_areas.get("stopping_criteria").setDatasetPlotTypeArtist("test_3_dataset", new VlinePlotArtist())

let figure2 = new Figure({
	container : document.getElementById("progress-plots"),
	shape : fig_shape,
	units : fig_units
})
figure2.appendPlotAreaFromRect("residual_histogram",  new R(0.05,0.05,0.9,0.9))
figure2.plot_areas.get("residual_histogram").appendDataAreaFromRect("residual_histogram_data_area", new R(0.1,0.1,0.8,0.8))
figure2.plot_areas.get("residual_histogram").appendAxesFromExtent("residual_histogram_axes", E.from(NaN,NaN,NaN,NaN), {axis_names : ["value", "count"], nonlinear_transform : log_transform_y})
figure2.plot_areas.get("residual_histogram").newDatasetForAxes("residual_histogram_axes", "residual_histogram_data")
figure2.plot_areas.get("residual_histogram").setDatasetPlotTypeArtist("residual_histogram_data", new StepPlotArtist())
figure2.plot_areas.get("residual_histogram").addDatasetToAxes("residual_histogram_axes", new RingbufferDataset("threshold_line_data",1))
figure2.plot_areas.get("residual_histogram").setDatasetPlotTypeArtist("threshold_line_data", new VlinePlotArtist())

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


plot_name_map.set("stopping_criteria", figure.plot_areas.get("stopping_criteria"))
plot_name_map.set("residual_histogram", figure2.plot_areas.get("residual_histogram"))
plot_name_map.set("component_histogram", figure3.plot_areas.get("component_histogram"))
plot_name_map.set("selected_pixels_histogram", figure4.plot_areas.get("selected_pixels_histogram"))
plot_name_map.set("current_convolved_histogram", figure5.plot_areas.get("current_convolved_histogram"))




let plt = plot_name_map.get("stopping_criteria")
plt.clear()
//plt.setCurrentDataset("test_1_dataset")
plt.addDataToDataset("test_1_dataset", 0,0)
plt.addDataToDataset("test_1_dataset", 1,1)
plt.addDataToDataset("test_1_dataset", 2,4)
//plt.addDataToDataset("test_1_dataset", 3,9)
//plt.addDataToDataset("test_1_dataset", 4,16)
//plt.addDataToDataset("test_1_dataset", 5,25)
//plt.addDataToDataset("test_1_dataset", 6,36)
//plt.addDataToDataset("test_1_dataset", 7,49)
//plt.addDataToDataset("test_1_dataset", 8,64)
plt.addDataToDataset("test_1_dataset", 9,80)
plt.addDataToDataset("test_1_dataset", 10,100)
plt.addDataToDataset("test_1_dataset", 11,121)
plt.addDataToDataset("test_1_dataset", 12,144)

console.log("Adding test data for 'test_2_dataset'")
plt.setAsDataset(
[[0,0.5*0],
[1,0.5*1],
[2,0.5*4],
[3,0.5*9],
[4,0.5*16],
[5,0.5*25],
[6,0.5*36],
[7,0.5*49],
[8,0.5*64],
[9,0.5*80],
[10,0.5*100],
[11,0.5*121],
[12,0.5*144],
],
"test_2_dataset"
)
/*
plt.addDataToDataset("test_2_dataset", 0,0.5*0)
plt.addDataToDataset("test_2_dataset", 1,0.5*1)
plt.addDataToDataset("test_2_dataset", 2,0.5*4)
plt.addDataToDataset("test_2_dataset", 3,0.5*9)
plt.addDataToDataset("test_2_dataset", 4,0.5*16)
plt.addDataToDataset("test_2_dataset", 5,0.5*25)
plt.addDataToDataset("test_2_dataset", 6,0.5*36)
plt.addDataToDataset("test_2_dataset", 7,0.5*49)
plt.addDataToDataset("test_2_dataset", 8,0.5*64)
plt.addDataToDataset("test_2_dataset", 9,0.5*80)
plt.addDataToDataset("test_2_dataset", 10,0.5*100)
plt.addDataToDataset("test_2_dataset", 11,0.5*121)
plt.addDataToDataset("test_2_dataset", 12,0.5*144)
*/
// Testing vertical line drawing and RingbufferDataset
//plt.addDataToDataset("test_3_dataset", 3,0)
//plt.addDataToDataset("test_3_dataset", 9,0)


download_clean_map_button.addEventListener(
	"click",
	new WasmDataDownloader( 
		()=>{
			let fn = ""
			let ext = ".tiff"
			if(sci_image_holder.file.name.endsWith(".tiff")){
				fn = sci_image_holder.file.name.slice(0,-5)+"_"
			}
			else if (sci_image_holder.file.name.endsWith(".tif")){
				fn = sci_image_holder.file.name.slice(0,-4)+"_"
				ext = ".tif"
			}
			
			return fn+"deconvolved_clean_map" + ext
		},
		"deconv.clean_map", 
		(file_id)=>Module.get_tiff(deconv_type, deconv_name, file_id, sci_image_holder.name),
		()=>{if(!deconv_complete){alert("Deconvolution not completed, cannot download results");} return deconv_complete;},
	)
)

download_residual_button.addEventListener(
	"click",
	new WasmDataDownloader( 
		()=>{
			let fn = ""
			let ext=".tiff"
			if(sci_image_holder.file.name.endsWith(".tiff")){
				fn = sci_image_holder.file.name.slice(0,-5)+"_"
			}
			else if (sci_image_holder.file.name.endsWith(".tif")){
				fn = sci_image_holder.file.name.slice(0,-4)+"_"
				ext = ".tif"
			}
			
			return fn+"deconvolved_residual"+ext
		},
		"deconv.residual", 
		(file_id)=>Module.get_tiff(deconv_type, deconv_name, file_id, sci_image_holder.name),
		()=>{if(!deconv_complete){alert("Deconvolution not completed, cannot download results");} return deconv_complete;},
	)
)

run_deconv_button.addEventListener("click", async (e)=>{
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
		
		console.log("run_deconv_button.addEventListener::click", Math.log10(clean_modified_params.valueOf("fabs_frac_threshold")))

		console.log("Preparing deconvolver for ${sci_image_holder.name} ${psf_image_holder.name}")
		await Module.prepare_deconvolver(deconv_type, deconv_name, sci_image_holder.name, psf_image_holder.name, "")
		
		
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
		width += (1-width%2)
		height += (1-height%2)

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
)





