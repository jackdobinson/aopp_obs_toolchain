

class WasmDataDownloader{
	constructor(
			filename, // <String> or <Callable(file_id)->String> Name of the file that will be downloaded, e.g., "image.jpg"
			file_id, // <Obj> argument passed to `wasm_data_provider` that enables retrieval of the data we want
			wasm_data_provider, // <Callable(Obj)->TypedArray> A function that provides the data for the download.
			predicate_required_for_download = ()=>true, // <Callable()->Bool> function called to see if download should proceed, true means download can proceed.
			on_success_callback = ()=>{}, //<Callable()->undefined> function to call when download is successful
			mime_type="application/octet-stream", // <String>
		){
		
		assert_all_defined(filename, file_id, wasm_data_provider, mime_type)

		this.filename = filename
		this.file_id = file_id
		this.wasm_data_provider = wasm_data_provider
		this.mime_type = mime_type
		this.predicate_required_for_download = predicate_required_for_download
		this.on_success_callback = on_success_callback
		this.resolved_filename = null
	}
	
	resolve_filename(){
		this.resolved_filename = typeof(this.filename) == "function" ? this.filename(this.file_id) : this.filename
	}
	
	unresolve_filename(){
		this.resolved_filename = null
	}
	
	handleEvent(){
		// This actually causes the download to start
		// Shamelessly copied from "https://gist.github.com/TrevorSundberg/74dc4f576b94d19711a4547dcf04af40"
		
		if(!this.predicate_required_for_download(this)){
			throw new Error("Predicate failed, cannot proceed with download")
		}
		
		
		let a = null
		let url = null
		let success = false
		
		try {
			a = document.createElement("a")
			a.style = 'display:none'
			document.body.appendChild(a)
			
			const wasm_data = this.wasm_data_provider(this.file_id)
			
			const blob = new Blob([wasm_data], {type:"octet/stream"})
			url = window.URL.createObjectURL(blob)
			
			a.href=url
			
			this.resolve_filename()
			a.download = this.resolved_filename
			a.click() // make the element perform its download
			
			success = true; // If we get here without throwing an error we are successful
		}
		catch (e){
			const msg = `An error occured during download. Recieved error message: ${e.message}`
			console.error(msg)
			alert(msg)
		}
		finally {
			this.unresolve_filename()
			if(url !== null){
				window.URL.revokeObjectURL(url)
			}
			if (a !== null){
				document.body.removeChild(a)
			}
		}
		
		if (success){
			this.on_success_callback()
		}
	}
}