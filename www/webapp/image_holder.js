


function getImageDataFromResult(get_result, arg_list, image_width, image_height){
	console.log(get_result, arg_list, image_width, image_height)
	let typed_array = get_result(...arg_list)
	console.log(typed_array)
	return new ImageData( new Uint8ClampedArray(typed_array.buffer, typed_array.byteOffset, typed_array.byteLength), image_width, image_height)
}


class ImageHolder {
	constructor(target_canvas, html_element, event_name, after_event_fn=()=>{}){
		this.target_canvas = target_canvas
		this.target_canvas_ctx = this.target_canvas.getContext("2d")
		this.after_event_fn = after_event_fn
		this.file=null
		this.name=null
		this.im_w=null
		this.im_h=null
		this.im_display_data=null

		if ((html_element !== undefined) && (event_name !== undefined)){
			html_element.addEventListener(event_name, this)
		}
	}

	discardPreviousImage(){
		if (this.name !== null){
			Module.remove_image(this.name)
		}
		this.file = null
		this.name = null
		this.im_w = null
		this.im_h = null
		this.im_display_data = null
	}

	async loadImageToModule(){
		if (this.file === null){
			console.log("ERROR: Cannot load a null file to module")
			return
		}

		if (this.file.name.endsWith(".tiff") || this.file.name.endsWith(".tif")) {
			this.name = Module.TIFF_from_js_array(this.file.name, await this.file.bytes())
		}
		else {
			console.log("ERROR: Unknown file extension. Cannot load image descr${this.file.name}.")
		}
	}

	getImageDimensions(){
		if(this.name === null){
			console.log("ERROR: Must have an image name to get image dimensions")
			return
		}
		this.im_w = Module.TIFF_get_width(this.name)
		this.im_h = Module.TIFF_get_height(this.name)
	}

	async displayImage(){
		if ((this.name===null) || (this.im_w===null) || (this.im_h===null)){
			console.log("ERROR: Must have image name, image width, and image height to display image")
			return
		}
		//this.im_display_data = new ImageData(new Uint8ClampedArray(Module.image_as_JSImageData(this.name)), this.im_w, this.im_h)
		this.im_display_data = await getImageDataFromResult(Module.image_as_JSImageData, [this.name], this.im_w, this.im_h)
		this.target_canvas.width = this.im_w
		this.target_canvas.height = this.im_h
		this.target_canvas.style = `aspect-ratio: ${this.im_w} / ${this.im_h};`
		this.target_canvas_ctx.imageSmoothingEnabled = false
		this.target_canvas_ctx.putImageData(this.im_display_data, 0, 0)

	}

	async handleEvent(e) {
		if (e.target.files.length != 1) {
			console.log("ERROR: Selected multiple files, only want a single file")
		}
		
		// update label show "LOADING..."
		for (const label of e.target.labels){
			let inner_html = label
			while (inner_html.firstElementChild !== null){
				inner_html = inner_html.firstElementChild
			}
			console.log(inner_html)
			inner_html.innerText= "LOADING..."
		}

		this.discardPreviousImage()

		this.file = e.target.files[0]
		let filename = this.file.name

		await this.loadImageToModule()

		this.getImageDimensions()
		this.displayImage()
		
		
		// update label to be the new file name
		for (const label of e.target.labels){
			let inner_html = label
			while (inner_html.firstElementChild !== null){
				inner_html = inner_html.firstElementChild
			}
			console.log(inner_html)
			inner_html.innerText=filename
		}
		
		this.after_event_fn()
	}
}


