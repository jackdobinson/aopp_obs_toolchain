
class AssertionError extends Error {
	constructor (message, options){
		super(message, options)
	}
}

function assert(bool_result, message=null){
	if (!bool_result){
		if (message===null){
			throw new AssertionError("Assertion failed!")
		} else {
			throw new AssertionError(message)
		}
	}
}

function not_null(obj){
	return ((obj!==undefined) && (obj!==null))
}

function assert_not_null(obj){
	assert(obj!==undefined, "object cannot be undefined")
	assert(obj!==null, "object cannot be null")
}

function assert_all_defined(...args){
	for(const [i,item] of args.entries()){
		assert(item !== undefined, `${i}th object was undefined`)
	}
}

function when_null(obj, value){
	if(!not_null(obj)){
		return value
	}
	return obj
}

function attrs_defined(obj, ...args){
	for (const item of args){
		assert(obj[item] !== undefined, `${obj} attribute ${item} cannot be undefined`)
	}
}

function attrs_not_null(obj, ...args){
	for (const item of args){
		//console.log(item)
		//console.log(obj[item])
		assert(obj[item] !== undefined, `${obj} attribute ${item} cannot be undefined`)
		assert(obj[item] !== null, `${obj} attribute ${item} cannot be null`)
	}
}

function attrs_satisfy(predicate, obj, ...args){
	for (const item of args){
		assert(obj[item] !== undefined, `${obj} attribute ${item} cannot be undefined`)
		assert(obj[item] !== null, `${obj} attribute ${item} cannot be null`)
		assert(predicate(obj[item]), `${obj} attribute ${item} must satisfy predicate`)
	}
}

class DataType{
	static next_id=0;
	
	constructor(
		name, // The name of the data type
		description, // a description of what the data type represented
		validator = (o)=>true // a function that takes an object and checks it is a valid member of the data type
	){
		this.id = DataType.next_id++ // unique ID for the data type
		this.name = name
		this.description = description
		this.validator = validator
	}
	
	validate(value){
		return this.validator(value)
	}
}

class TypeRegistry{
	static id_map = new Map();
	static name_map = new Map();
	
	static add(...data_types){
		for(const data_type of data_types){
			attrs_not_null(data_type, 'id', 'name', 'description')
			assert(!TypeRegistry.id_map.has(data_type.id))
			assert(!TypeRegistry.name_map.has(data_type.name))
			TypeRegistry.id_map.set(data_type.id, data_type)
			TypeRegistry.name_map.set(data_type.name, data_type)
		}
	}
	
	static has(name_or_id){
		let present = false
		if (typeof(name_or_id)=="number"){
			present = TypeRegistry.id_map.has(name_or_id)
		} 
		else if (typeof(name_or_id)=="string"){
			present = TypeRegistry.name_map.has(name_or_id)
		}
		return present
	}
	
	static get_by_name(name){
		assert(not_null(name))
		return TypeRegistry.name_map.get(name)
	}
	
	static get_by_id(id){
		assert(not_null(id))
		return TypeRegistry.id_map.get(id)
	}
}

TypeRegistry.add(
	new DataType("integer", "A whole number between -inf and +inf", (o)=>((typeof(o)=="number") && Number.isInteger(o))),
	new DataType("integer(1,inf)", "A whole number between 1 and +inf", (o)=>((typeof(o)=="number") && Number.isInteger(o) && (o>=1))),
	new DataType("integer(0,inf)", "A whole number between 0 and +inf", (o)=>((typeof(o)=="number") && Number.isInteger(o) && (o>=0))),
	new DataType("real", "A real number between -inf and +inf", (o)=>(typeof(o)=="number")),
	new DataType("real(0,1)", "A real number in the interval (0, 1)", (o)=>((typeof(o)=="number")&&(0<o)&&(o<1))),
	new DataType("bool", "A boolean that can be one of {true, false}", (o)=>(typeof(o)=="boolean")),
)


class Parameter{
	static next_id=0;
	
	constructor(
		name, // the name of the parameter
		description, // a description explaining what the paramter controls
		type_name, // the name of the data type of the parameter, must be one of the registered data types
		deserialiser, // function(string) -> type // When given a string that represents a value, sets the value of the parameter
		default_value = null, // default value to use when no other value is provided. 
		validator = (v)=>true, // function(type) -> bool // ensures parameter value is in correct range
		serialier = (v)=>{return v.toString()}, // function(type) -> string // represents the value of the parameter as a string
	){
		this.id = Parameter.next_id++
		this.name = name
		this.description = description
		assert(TypeRegistry.has(type_name))
		this.type_name = type_name
		this.validator = validator
		this.default_value = default_value
		this.deserialiser = deserialiser
		this.serialier = serialier
	}
	
	validate(value){
		let is_validated = true
		is_validated &&= TypeRegistry.get_by_name(this.type_name).validate(value)
		is_validated &&= this.validator(value)
		return is_validated
	}
	
	deserialise(value){
		return this.deserialiser(value)
	}
}


class Control{
	constructor(html_container, input_element, label_element, value_getter, deserialiser, validator){
		this.html_container = html_container
		this.input_element = input_element
		this.label_element = label_element
		this.value_getter = value_getter
		this.deserialiser = deserialiser
		//console.log("xxx", validator)
		this.validator = validator
	}
	
	addEventListener(type, listener, ...args){
		return this.input_element.addEventListener(type, listener, ...args)
	}
	
	dispatchEvent(event){
		return this.input_element.dispatchEvent(event)
	}
	
	getValue(){
		return this.deserialiser(this.value_getter(this.input_element))
	}
	
	validate(){
		return this.validator(this.getValue())
	}
}

class ControlManager{
	static next_id = 0
	
	static set_attributes_of(html_element, attributes){
		if (attributes['id']===undefined){
			attributes["id"] = `auto-id-${ControlManager.next_id++}`
		}
	
		for(const attr of Object.keys(attributes)){
			html_element.setAttribute(attr, attributes[attr])
		}
		return html_element
	}
	
	static create_container(attributes){
		let html_element = document.createElement('div')
		return ControlManager.set_attributes_of(html_element, attributes)
	}
	
	static create_control(type, attributes, value_getter, deserialiser, validator){
		let html_element = document.createElement('div')
		ControlManager.set_attributes_of(html_element, {class:'param-control'})
		
		let input_element = document.createElement('input')
		attributes["type"] = type
		if (attributes["class"] === undefined){
			attributes["class"] = "param-control-input"
		} 
		else {
			attributes["class"] += " param-control-input"
		}
		ControlManager.set_attributes_of(input_element, attributes)
		
		
		let label_element = document.createElement('label')
		ControlManager.set_attributes_of(label_element, {class:"param-control-label", for:input_element.id})
		
		html_element.append(label_element, input_element)
		
		return new Control(html_element, input_element, label_element, value_getter, deserialiser, validator)
	}
	
	static create_control_for(param, on_valid_fn, on_invalid_fn){
		let ctl = null
		let input_type = null
		
		//console.log(param)
		
		switch(param.type_name){
			case "integer(0,inf)":
				input_type = "number"
				ctl = ControlManager.create_control(
					input_type,
					{id:param.name, min:0, step:1, value:when_null(param.default_value,0), class:`param-${param.type_name}`}, 
					(x)=>x.value, 
					param.deserialiser, 
					param.validate.bind(param)
				)
				break
			case "integer(1,inf)":
				input_type = "number"
				ctl = ControlManager.create_control(
					input_type,
					{id:param.name, min:1, step:1, value:when_null(param.default_value,1), class:`param-${param.type_name}`}, 
					(x)=>x.value, 
					param.deserialiser, 
					param.validate.bind(param)
				)
				break
			case "integer":
				input_type = "number"
				ctl = ControlManager.create_control(
					input_type,
					{id:param.name, step:1, value:when_null(param.default_value,0), class:`param-${param.type_name}`}, 
					(x)=>x.value, 
					param.deserialiser, 
					param.validate.bind(param)
				)
				break
			case "bool":
				input_type = "checkbox"
				ctl = ControlManager.create_control(
					input_type, 
					{id:param.name, value:param.name, class:`param-${param.type_name}`}, 
					(x)=>x.checked, 
					param.deserialiser, 
					param.validate.bind(param)
				)
				if (not_null(param.default_value) && param.default_value){
					ctl.input_element.checked=true
				} else {
					ctl.input_element.checked=false
				}
				
				break
			case "real(0,1)":
				input_type = "number"
				ctl = ControlManager.create_control(
					input_type, 
					{id:param.name, min:0, max:1, step:0.01, value:when_null(param.default_value,0.5), class:`param-${param.type_name}`}, 
					(x)=>x.value, 
					param.deserialiser, 
					param.validate.bind(param)
				)
				break
			case "real":
				input_type = "number"
				ctl = ControlManager.create_control(
					input_type, 
					{id:param.name, value:when_null(param.default_value,0), class:`param-${param.type_name}`}, 
					(x)=>x.value, 
					param.deserialiser,
					param.validate.bind(param)
				)
				break
			default:
				assert(false, `Unknown param type '${param.type_name}' to make control for`)
		}
		
		//console.log(ctl)
		
		assert_not_null(ctl)
		
		// Set tooltips for controls
		ctl.html_container.classList.add("has-tooltip")
		
		ctl.label_element.textContent = param.name
		
		let tooltip_element = document.createElement("p")
		tooltip_element.classList.add("tooltip")
		tooltip_element.textContent = param.description
		
		ctl.html_container.appendChild(tooltip_element)
		ctl.tooltip_element = tooltip_element
		
		// Set validator tips
		let validator_tip_element = document.createElement("p")
		validator_tip_element.classList.add("validator-tip")
		validator_tip_element.textContent = "Value should be " + TypeRegistry.get_by_name(param.type_name).description.toLowerCase()
		
		ctl.html_container.appendChild(validator_tip_element)
		ctl.validator_tip_element = validator_tip_element
		
		// Set validators for controls
		ctl.html_container.classList.add("has-validator")
		ctl.html_container.setAttribute("validated", "unknown")
		ctl.addEventListener("change", (e)=>{
				if (ctl.validate()){
					ctl.html_container.setAttribute("validated", "true")
					on_valid_fn()
				} else {
					ctl.html_container.setAttribute("validated", "false")
					on_invalid_fn()
				}
			}
		)
		
		return ctl
	}
}

class CleanModifiedParameters{
	static parameters = [
		new Parameter(
			'n_iter', 
			'Maximum number of iterations to perform. A good starting number is 100', 
			'integer(1,inf)', 
			Number, 
			100
		),
		new Parameter(
			"adaptive_threshold_flag", 
			"If true, will apply heuteristics to find the optimal threshold at each iteration and not rely upon a manually set threshold", 
			"bool", 
			Boolean, 
			true
		),
		new Parameter(
			'threshold', 
			"Fraction of the residual's brightest pixel, above which a pixel will be selected as a 'source pixel'. Must be in the range (0,1).", 
			"real(0,1)", 
			Number, 
			0.3
		),
		new Parameter(
			"loop_gain", 
			"What fraction of a selected pixel is treated as a 'source' each iteration. Must be in the range (0,1).", 
			"real(0,1)", 
			Number, 
			0.1
		),
		new Parameter(
			"rms_frac_threshold", 
			"When the root-mean-square of the residual is below this fraction of it's original value, iteration will stop. Must be in the range (0,1).", 
			"real(0,1)", 
			Number, 
			1E-2
		),
		new Parameter(
			"fabs_frac_threshold", 
			"When the brightest pixel of the residual is below this fraction of it's original value, iteration will stop. Must be in the range (0,1).", 
			"real(0,1)", 
			Number, 
			1E-2
		),
		new Parameter(
			"clean_beam_sigma", 
			"The standard deviation (in pixels) of the gaussian 'clean beam' to convolve source components with, forming the 'clean map'. If zero, no clean beam is used", 
			"real", 
			Number, 
			0
		),
		new Parameter(
			"add_residual_flag", 
			"If true, will add the residual to the clean map after convolution with the cleam beam (if requested)", 
			"bool", 
			Boolean, 
			false
		),
		new Parameter(
			"plot_update_interval", 
			"How often to update plots (in number of iterations). Less frequent updates are faster, set to zero to never update plots.", 
			"integer(0,inf)", 
			Number, 
			10
		),
	];
	
	constructor(parent_element){
		this.ctl_container = ControlManager.create_container({class:"param-container"})
		
		this.parameter_ctls = new Map()
		
		// build controls for each parameter
		for (const param of CleanModifiedParameters.parameters){
			this.parameter_ctls.set(
				param.name,  
				ControlManager.create_control_for(
					param,
					()=>{
						let is_good = this.validate().length == 0
						deconv_status_mgr.set("Parameters Validated", is_good, {"is-good":is_good})
					},
					()=>{
						deconv_status_mgr.set("Parameters Validated", false, {"is-good":false})
					}
				)
			)
		}
	
	
		// Create groups of parameters that influence reported values
		
		// add parameter controls to correct place in html document
		for(const [key, value] of this.parameter_ctls){
			this.ctl_container.append(value.html_container)
		}

		// Add event listeners here

		this.parameter_ctls.get("adaptive_threshold_flag").addEventListener("change", (e)=>{
				this.parameter_ctls.get("threshold").input_element.disabled = e.target.checked
			}
		)
		
		// want to run this on initialisation as well as when control is changed
		this.parameter_ctls.get("adaptive_threshold_flag").dispatchEvent(new Event("change"))
		//this.adaptive_threshold_flag_ctl.dispatchEvent(new Event("change"))
		
		parent_element.appendChild(this.ctl_container)
	}
	
	validate(){
		let invalid_params = []
		for(const [key, param_ctl] of this.parameter_ctls){
			if(!param_ctl.validate()){
				invalid_params.push(key)
			}
		}
		return invalid_params
	}
	
	values(){
		let param_ctl_values = new Map()
		for(const [key, param_ctl] of this.parameter_ctls){
			param_ctl_values.set(key, param_ctl.getValue())
		}
		return param_ctl_values
	}
	
	valueOf(param_name){
		this.parameter_ctls.get(param_name).getValue()
	}
	
	
	set_params(deconv_type, deconv_name){
		console.log(deconv_type, deconv_name)
		
		let invalid_params = this.validate()
		
		if(invalid_params.length != 0){
			return invalid_params
		}
		
		let param_ctl_values = this.values()
		for(const [key, value] of param_ctl_values){
			console.log(`Parameter '${key}' = ${value}`)
		}
		
		Module.set_deconvolver_parameters(
			deconv_type, 
			deconv_name,
			param_ctl_values.get("n_iter"),
			0, //n_positive_iter
			param_ctl_values.get("loop_gain"),//this.loop_gain_ctl.getValue(),
			param_ctl_values.get("adaptive_threshold_flag"),//this.adaptive_threshold_flag_ctl.getValue(),
			param_ctl_values.get("threshold"),//this.threshold_ctl.getValue(),
			param_ctl_values.get("clean_beam_sigma"),//this.clean_beam_sigma_ctl.getValue(),
			param_ctl_values.get("add_residual_flag"),//this.add_residual_flag_ctl.getValue(),
			1E-2,
			param_ctl_values.get("rms_frac_threshold"),//this.rms_frac_threshold_ctl.getValue(),
			param_ctl_values.get("fabs_frac_threshold"),//this.fabs_frac_threshold_ctl.getValue()
			param_ctl_values.get("plot_update_interval"),
		)
		
		return invalid_params // need to return a list
	}
}