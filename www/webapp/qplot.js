
// TODO:
// * Have a think about breaking up the dependence of DataArea and Axis upon eachother
// * Want to have some way to draw arbitrary lines on plots e.g. "axvline" from matplotlib
// * Ideally be able to hot-swap between log and non-log axes (plus others if possible)
// * Would be nice to be able to have more than one set of axes on a DataArea
// * Would be nice to plot multiple datasets in their own colour
// * Would be nice to be able to set styles etc. of lines, text, markers etc.


/*
Anatomy of a plot:

Figures hold plot areas
plot areas hold axes and data areas
data areas hold data points

##############################################
# figure                                     #
#                                            #
#    ---------------------------             #
#    | plot_area               |             #
#    |                         |             #
#    |    |axes                |             #
#    |    V                    |             #
#    |   ##   ...........      |             #
#    |   ##   . data    .      |             #
#    |   ##   . Area    .      |             #
#    |   ##   .         .      |             #
#    |   ##   ...........      |             #
#    |   ##                    |             #
#    |   ################      |             #
#    |   ################      |             #
#    |                         |             #
#    ---------------------------             #
#                                            #
#                                            #
##############################################

"display coordinates" are in terms of whatever we are drawing on, e.g. a screen or a canvas. This is
the coordinate system that we ultimately convert to.

"figure coordinates" always run from 0 to 1, with (0,0) at the bottom left and (1,1) at the top right.

"plot_area coordinates" always run from 0 to 1, with (0,0) at the bottom left and (1,1) at the top right.

dataArea represents the area of a plot that we want to
draw a dataset inside. Can be any geometrical shape but usually a rectangle

A dataset is a collection of data. I.e. the (x,y) coords of point data, 
the (category, count) values of categorical data. All data in a dataset
MUST share the same interpretation. I.e. cannot mix (x,y) data with 
(category, count) data within a dataset.

When a dataset is drawn to a dataArea, it's values are transformed from
its "dataset coordinates" to "representation coordinates" to "dataArea coordinates" 

"dataArea coordinates" always run from 0 to 1, with (0,0) at the bottom left and (1,1) at the top right.

"representation coordinates" are an intermediate coordinate system that defines how the axes are drawn.
This allows us to e.g. plot the logarithm of a dataset instead of the dataset itself just by changing 
the "representation coordinate transform". Data that is transformed to the same "representation coordinates"
can SHARE and axis.

The "dataset coordinates" are the raw values of the data in the dataset. In simple cases the transformation
from "dataset coordinates" to "representation coordinates" is the identity transform. However, if we
e.g. have two datasets (A and B) where A is a collection of (time, instrument_A_reading), and B is a collection
of (time, instrument_B_reading), and we know that instrument_A_reading is in units X, but instrument_B_reading is
in units log(X). We can plot them on the same axis by letting the dataset2representation_transform for each dataset be
datasetA2representation_transform = identity, datasetB2representation_transform=(b[0],exp(b[1])).

Therefore, each dataset requires its own dataset2representation_transform.

These coord systems are defined in terms of transforms:
* dataset2representation_transform
* representation2dataArea_transfrom
* dataArea2plotArea_transform
* plotArea2figure_transform
* figure2display_transform

An axis is a visualisation of a "representation coordinate" system. It is placed in plot coordinates,
usually not covering the dataArea. One axis represents the change of one component of a "representation coordinate" system
with the other components held constant. This is usually visuallised as a solid line (the constant part in a 2d coord system),
with tick-marks and tick-labels along its length (marking the varying part in a 2d coord system).

An axis can be placed anywhere, but is best placed at the edges of the dataArea its "representation coordinate" system corresponds to.

A "representation coordinate" system can have multiple visualisations if needed, for example gridding over the dataArea is
another visualisation of a "representation coordinate" system.

Annotations on a plot are things like drawing a box around a region, vertical lines to indicate something important, text to
highlight specific datapoints or label datasets etc. Depending on the annotation they will usually be defined in either the
"dataset", "representation", "dataArea", or sometimes "plotArea" coordinate systems.

NOTE: with text annotations, it is often important to define their position in "dataset" coords, but to draw the characters in
"plotArea", "figure" or even "display" coords. This is because generally we do not want to apply transforms to the shapes of the
character glyphs, we only want to apply transforms to where the text is placed.

*/



class Dataset{
	constructor(
		name
	){
		this.name = name
		this.storage = [] // simplest for now
		this.read_heads = new Map() // index of next data to read for axes that read data from dataset
	}
	
	clear(){
		this.storage = []
		for(const [axes_name, read_head_idx] of this.read_heads.entries()){
			this.read_heads.set(axes_name, 0)
		}
	}
	
	set(data){
		this.storage = data
	}
	
	push(data){
		this.storage.push(data)
	}
	
	resetIterator(axes_name){
		//console.log("Dataset::resetIterator", axes_name)
		this.read_heads.set(axes_name,0)
	}
	
	*getNewData(axes_name){
		let current_index = this.read_heads.get(axes_name)
		if (current_index === undefined){
			current_index = 0
		}
		
		for(; current_index < this.storage.length; current_index++){
			yield this.storage[current_index]
		}
		this.read_heads.set(axes_name, current_index)
	}
	
	*getData(axes_name){
		let current_index = 0
		for(; current_index < this.storage.length; current_index++){
			yield this.storage[current_index]
		}
		this.read_heads.set(axes_name, current_index)
	}
}

class RingbufferDataset extends Dataset {
	constructor(
		name,
		n_max_entries = 1,
	){
		super(name)
		this.n_max_entries = n_max_entries
		this.write_head = 0
		this.loop_around = 0
	}
	
	clear(){
		super.clear()
		this.write_head = 0
		this.loop_around = 0
	}
	
	set(data){
		if (data.length <= n_max_entries){
			super.set(data)
			this.loop_around = 0
			this.write_head = data.length
		}
		else {
			this.storage = new Array(this.n_max_entries)
			let n_from_final_write = (data.length % this.n_max_entries)
			this.storage.splice(0,0,...data.slice(data.length-n_from_final_write, n_from_final_write))
			this.storage.splice(n_from_final_write, 0, ...data.slice(data.length-this.storage.length, data.length-n_from_final_write))
			this.loop_around = 1
			this.write_head = n_from_final_write
		}
	}
	
	push(data){
		if(this.storage.length == this.n_max_entries){
			this.storage[this.write_head] = data
			this.write_head = (this.write_head + 1) % this.n_max_entries
			this.loop_around = 1
		} else {
			this.storage.push(data)
			this.write_head++
		}
	}
	
	*getNewData(axes_name){
		let current_index = this.read_heads.get(axes_name)
		if (current_index === undefined){
			current_index = 0 + this.loop_around*this.write_head
		}
		
		let i = 0
		let read_loop = current_index < this.write_head
		let n_to_read = this.write_head - current_index
		
		if(read_loop){
			n_to_read += this.loop_around*this.n_max_entries
		}		
		while(i < n_to_read){
			yield this.storage[current_index]
			current_index = (current_index + 1) % this.n_max_entries
			i++
		}
		
		this.read_heads.set(axes_name, current_index)
	}
	
	*getData(axes_name){
		if (this.loop_around == 0){
			yield* super.getData(axes_name)
		}
		else {
			let i = 0
			let current_index = this.write_head
			while(i < this.n_max_entries){
				yield this.storage[current_index]
				current_index = (current_index + 1) % this.n_max_entries
				i++
			}
			this.read_heads.set(axes_name, current_index)
		}
	}
}


class PlotTypeArtist{
	// This object is responsible for creating the SVG representation of a dataset
	// this should be subclassed for different types of plot
	// The default implementation is a line graph
	
	static plot_style_defaults = {
		"marker-type" : "none",
		"line-type" : "solid", // "solid", "dash", "dot", any combination of "dash" and "dot" separated by "-" e.g. "dot-dash-dot"
	}
	
	static line_style_defaults = {
		"stroke-width" : 3,
		"stroke" : null,
		"fill" : "none",
		"stroke-opacity" : 0.6,
		"fill-opacity" : 0.6,
		"vector-effect" : "non-scaling-stroke",
	}
	
	static marker_style_defaults = {
		"stroke-width" : 3,
		"stroke" : "none",
		"fill" : null,
		"stroke-opacity" : 0.6,
		"fill-opacity" : 0.6,
		"vector-effect" : "non-scaling-stroke",
	}
	
	static colour_progression = [
		"blue",
		"red",
		"green",
		"purple",
		"brown",
		"orange",
	]
	
	static n_instances = 0
	
	constructor({
		name = null,
		classname = null, // sets the class of the containing group, useful for selecting with CSS
		plot_style = {},
		line_style = {},
		marker_style = {},
	} = {}){
		PlotTypeArtist.n_instances++
	
		this.name = (name===null) ? `plot-type-artist-${PlotTypeArtist.n_instances}` : name
		this.needs_data_init = true
		this.default_primary_colour = PlotTypeArtist.colour_progression[(PlotTypeArtist.n_instances % PlotTypeArtist.colour_progression.length)]
		
		this.setStyles(plot_style, line_style, marker_style)
		
		this.svg_parent_to_child_map = new Map()
		this.svg = new SvgContainer(
			Svg.group(`group-${this.name}`, null, {"class":classname})
		)
		this.data_svg = new SvgContainer(
			this.svg.add('group',`group-data-${this.name}`, null, {"class":"datapoints"})
		)
	}
	
	setStyles(plot_style, line_style, marker_style){
		this.setPlotStyle(plot_style)
		this.setLineStyle(line_style)
		this.setMarkerStyle(marker_style)
	}
	
	setPlotStyle(...args){
		//console.log("PlotTypeArtist::setPlotStyle() ", ...args, PlotTypeArtist.plot_style_defaults)
		//console.log("this.constructor.plot_style_defaults", this.constructor.plot_style_defaults)
		//console.log()
		this.plot_style = O.insertIfNotPresent(...args, O.getStaticAttrOf(this, "plot_style_defaults"))
	}
	
	setLineStyle(...args){
		//console.log("PlotTypeArtist::setLineStyle() ", ...args, PlotTypeArtist.plot_style_defaults)
		this.line_style = O.insertIfNotPresent(...args, O.getStaticAttrOf(this, "line_style_defaults"))
		
		let stroke_width = this.line_style["stroke-width"]
		assert(stroke_width !== undefined, "Must have a stroke width defined")
		
		let add_styles = {}
		if(this.plot_style["line-type"] != "solid"){
			let dasharray = ""
			for(const word of this.plot_style["line-type"].split("-")){
				switch(word){
					case "dot":
						dasharray += `${1*stroke_width} ${1*stroke_width} ` // "dash_size gap_size"
						break
					case "dash":
						dasharray += `${3*stroke_width} ${1*stroke_width} ` // "dash_size gap_size"
						break
					default:
						throw Error(`Unrecognised line-type dasharray specifier '${word}' in line-type string '${this.plot_style["line-type"]}'`)
						break
				}
			}
			add_styles["stroke-dasharray"] = dasharray
		}
		this.line_style = O.insertIfNotPresent(this.line_style, add_styles)
		if (this.line_style["stroke"] === null){
			this.line_style["stroke"] = this.default_primary_colour
		}
		if (this.line_style["fill"] === null){
			this.line_style["fill"] = this.default_primary_colour
		}
	}
	
	setMarkerStyle(...args){
		//console.log("PlotTypeArtist::setMarkerStyle() ", ...args, PlotTypeArtist.plot_style_defaults)
		this.marker_style = O.insertIfNotPresent(...args, O.getStaticAttrOf(this, "marker_style_defaults"))
		if (this.marker_style["stroke"] === null){
			this.marker_style["stroke"] = this.default_primary_colour
		}
		if (this.marker_style["fill"] === null){
			this.marker_style["fill"] = this.default_primary_colour
		}
	}
	
	clear(){
		this.data_svg.clear()
		this.needs_data_init = true
	}
	
	addSvgTo(parent){
		this.svg_parent_to_child_map.set(parent, parent.appendChild(this.svg.root))
	}
	
	removeSvgFrom(parent){
		let child_node = this.svg_parent_to_child_map.get(parent)
		if(child_node !== undefined){
			parent.removeChild(this.svg_parent_to_child_map.get(parent))
			this.this.svg_parent_to_child_map.delete(parent)
		}
	}
	
	drawDataWithTransform(data, toRoot_transform){
		// data is in DataArea coords at this point
		//console.log("PlotTypeArtist::drawDataWithTransform() toRootTransform", toRoot_transform)
		this.drawData(T.apply(toRoot_transform, data))
	}
	
	drawData(data){
		// data is is root coords at this point
		//console.log("PlotTypeArtist::drawData is abstract method")
	}
	
	drawDataIterable(data_iterable){
		for(const item of data_iterable){
			this.drawData(item)
		}
	}
	
}


class LinePlotArtist extends PlotTypeArtist {

	static plot_style_defaults = {
		"marker-type" : "circle"
	}

	static marker_style_defaults = {
		"stroke" : "none",//"context-stroke", // "none"
		"fill" : null,//"context-fill", //colour,
		"fill-opacity" : 0.6,
		"stroke-width" : 0.3,
		"stroke-opacity" : 0.6,
	}
	
	static line_style_defaults = {
		"marker-start" : null,
		"marker-mid" : null,
		"marker-end" : null,
		"stroke" : null,
		"stroke-width" : 5,
		"stroke-opacity" : 0.6,
		"fill" : "none",
		"fill-opacity" : 0.6,
		"vector-effect" : "non-scaling-stroke",
	}

	constructor({
		name = null,
		classname = null,
		plot_style = {},
		marker_style = {},
		line_style = {},
	}={}){
		//console.log("LinePlotArtist::constructor()")
		super({name:name, classname:classname, plot_style:plot_style, line_style:line_style, marker_style:marker_style,})
		
		this.default_marker_name = `marker-${this.name}`
		this.create_default_marker = false
	
	
		this.defs = new SvgContainer(this.svg.add("defs"))
		
		
		if(this.create_default_marker){
			this.createMarker()
		}
		
		this.createLine()
	}
	
	createLine(){
		this.needs_data_init = false
		this.line = this.data_svg.add(
			"polyline",
			"",
			this.line_style
		)
	}
	
	setMarkerStyle(...args){
		super.setMarkerStyle(...args)
		
		if(this.marker_style.id === undefined){
			this.marker_style.id = this.default_marker_name
		}
	}
	
	setLineStyle(...args){
		super.setLineStyle(...args)
		
		if (this.line_style["marker-start"] === null){
			this.line_style["marker-start"] = `url(#${this.default_marker_name})`
			this.create_default_marker = true
		}
		if (this.line_style["marker-mid"] === null){
			this.line_style["marker-mid"] = `url(#${this.default_marker_name})`
			this.create_default_marker = true
		}
		if (this.line_style["marker-end"] === null){
			this.line_style["marker-end"] = `url(#${this.default_marker_name})`
			this.create_default_marker = true
		}
	}
	
	createMarker(){
		this.default_marker = new SvgContainer(
			this.defs.add(
				"marker", 
				{
					id : this.default_marker_name,
					viewBox : "-1 -1 2 2",
					markerWidth : 3,
					markerHeight: 3,
					markerUnits : "strokeWidth",
					orient : 0, //"auto",//"auto-start-reverse",
					refX : 0,
					refY: 0,
				}
			)
		)
		
		switch (this.plot_style["marker-type"]) {
			case "circle":
				this.default_marker.add(
					"circle", 
					V.zeros(2),
					1,
					this.marker_style
				)
				break
			default:
				throw Error(`Unknown marker type "${this.marker_type}" requested`)
		}
	}
	
	drawData(data){	
		// May have to swap this for a DOMPoint at some time in the future
		if (this.needs_data_init){
			this.createLine()
		}
		
		//console.log(this.line)
		
		let p = this.line.ownerSVGElement.createSVGPoint()
		//console.log(data[0], data[1])
		//console.log(typeof(data[0]), typeof(data[1]))
		p.x = data[0]
		p.y = data[1]
		this.line.points.appendItem(p)
		
	}
}

class StepPlotArtist extends PlotTypeArtist {
	
	static plot_style_defaults = {
		"step-pos" : 0, // Where do we step from one value to another. 0 - right after the previous datapoint, 1 - right before the current datapoint, 0.5 - mid-way between datapoints
		"extend-left" : false,
		"extend-right" : false,
	}
	
	static line_style_defaults = {
		"stroke" : null,
		"stroke-width" : 1,
		"stroke-opacity" : 0.6,
		"fill" : "none",
		"fill-opacity" : 0.6,
	}
	
	static marker_style_defaults = {}

	constructor({
		name = null,
		classname = null,
		plot_style = {},
		marker_style = {},
		line_style = {},
	}={}){
		//console.log("StepPlotArtist::constructor()")
		super({name:name, classname:classname, plot_style:plot_style, line_style:line_style, marker_style:marker_style,})		
	
		//console.log(Object.getPrototypeOf(this))
		//console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(this)))
		//console.log(Object.getOwnPropertyNames(this.constructor))
	
		this.default_marker_name = `marker-${this.name}`
		this.create_default_marker = false
	
		this.defs = new SvgContainer(this.svg.add("defs"))
		
		this.adjust_first_point = false
		this.previous_point = null
		this.createLine()
	}
	
	clear(){
		//console.log("StepPlotArtist::clear()")
		super.clear()
		this.previous_point = null
		this.adjust_first_point = false
	}
	
	createLine(){
		this.needs_data_init = false
		this.line = this.data_svg.add(
			"polyline",
			"",
			this.line_style
		)
	}
	
	drawData(data){	
		//console.log("StepPlotArtist::drawData()", data, this.previous_point)
		//console.trace()
		// May have to swap this for a DOMPoint at some time in the future
		if (this.needs_data_init){
			//console.log("StepPlotArtist::drawData() initialising data drawing")
			this.createLine()
		}
		
		//console.log(this.line)
		let p_last_value = this.line.ownerSVGElement.createSVGPoint()
		let p_this_value = this.line.ownerSVGElement.createSVGPoint()
		
		if(this.previous_point === null){
			//console.log("Previous point is null, writing first datapoint")
			
			p_last_value.x = data[0]
			p_last_value.y = data[1]
			this.line.points.appendItem(p_last_value)
			
			if(this.plot_style["extend-right"]){
				p_this_value.x = data[0]
				p_this_value.y = data[1]
				this.line.points.appendItem(p_this_value)
			}
			
			this.previous_point = V.from(data[0],data[1])
			this.adjust_first_point = true
			
			//console.log(this.line)
			//console.log(this.line.points)
		} 
		else {
			//console.log("Previous point is not null, continuing step plot.")
			//console.log(this.plot_style)
			let delta_x = this.plot_style["step-pos"]*(data[0] - this.previous_point[0])
			
			//console.log("delta_x", delta_x)
			
			
			if(this.plot_style["extend-left"] && this.adjust_first_point){
				this.line.points[0].x = this.previous_point[0] - (1-this.plot_style["step-pos"])*(data[0] - this.previous_point[0])
				this.adjust_first_point = false
			}
			
			if(this.plot_style["extend-right"]){
				this.line.points[this.line.points.length-1].x = this.previous_point[0] + delta_x
				
				p_last_value.x = this.previous_point[0] + delta_x
				p_last_value.y = data[1]
				this.line.points.appendItem(p_last_value)
				
				p_this_value.x = data[0] + delta_x
				p_this_value.y = data[1]
				this.line.points.appendItem(p_this_value)
			} 
			else {
				p_last_value.x = this.previous_point[0] + delta_x
				p_last_value.y = this.previous_point[1]
				this.line.points.appendItem(p_last_value)
				
				p_this_value.x = this.previous_point[0] + delta_x
				p_this_value.y = data[1]
				this.line.points.appendItem(p_this_value)
			}
			
			this.previous_point[0] = data[0]
			this.previous_point[1] = data[1]
		}
		
		//console.log(this.line)
	}
}

class VlinePlotArtist extends PlotTypeArtist {
	static plot_style_defaults = {}
	static line_style_defaults = {}
	static marker_style_defaults = {}
	
	constructor({
		name = null,
		classname = null,
		plot_style = {},
		marker_style = {},
		line_style = {},
	}={}){
		//console.log("VlinePlotArtist::constructor()")
		super({name:name, classname:classname, plot_style:plot_style, line_style:line_style, marker_style:marker_style,})
		
		this.lines = new Array()
	}
	
	drawDataWithTransform(data, toRoot_transform){
		//console.log("VlinePlotArtist::drawDataWithTransform()", data)
		let p0 = T.apply(toRoot_transform, [data[0],0])
		let p1 = T.apply(toRoot_transform, [data[0],1])
		this.drawData([p0[0],p0[1],p1[0],p1[1]])
	}
	
	drawData(data){
		// only take the x axis value, draw from 0-1 in the y axis
		
		//console.log("VlinePlotArtist::drawData()", data)
		
		this.lines.push(
			this.data_svg.add(
				"line",
				data,
				this.line_style
			)
		)
	}
}


class ReferenceFrame{
	
	static fromRect(parent_frame, this_rect_in_parent){
		return new ReferenceFrame(
			parent_frame,
			R.getTransformFromUnitCoordsTo(this_rect_in_parent)
		)
	}
	
	static fromRectReverse(parent_frame, this_rect_in_parent){
		return new ReferenceFrame(
			parent_frame,
			R.getTransformToUnitCoordsFrom(this_rect_in_parent)
		)
	}
	
	static fromExtent(parent_frame, this_extent_in_parent){
		//console.log("this_extent_in_parent", this_extent_in_parent)
		return ReferenceFrame.fromRect(parent_frame, R.fromExtent(this_extent_in_parent))
	}

	static fromExtentReverse(parent_frame, this_extent_in_parent){
		//console.log("this_extent_in_parent", this_extent_in_parent)
		return ReferenceFrame.fromRectReverse(parent_frame, R.fromExtent(this_extent_in_parent))
	}
	
	constructor(
		parent_frame = null,
		toParent_transform = T.identity,
		child_frames = []
	){
		assert(parent_frame !== undefined, "parent frame cannot be undefined, but may be null")
		this.parent_frame = parent_frame
		//console.log("this.parent_frame", this.parent_frame)
		
		
		
		// transform to previous reference frame
		this.toParent_transform = toParent_transform
		this.child_frames = child_frames
		
		this.toRoot_transform = null
		this.updateRootTransform()
	}
	
	updateRootTransform(){
		this.toRoot_transform = T.prod(
			this.parent_frame === null ? T.identity : this.parent_frame.toRoot_transform, 
			this.toParent_transform
		)
	}
	
	addChild(fromChild_transform){
		this.child_frames.push(new ReferenceFrame(this, fromChild_transform))
	}
	
	addChildFromRect(child_rect_in_this){
		this.child_frames.push(ReferenceFrame.fromRect(this, child_rect_in_this))
	}
	
	addChildFromExtent(child_extent_in_this){
		this.child_frames.push(ReferenceFrame.fromExtent(this, child_extent_in_this))
	}
	
}


class NonlinearTransform{
	constructor(forward_fn, backward_fn){
		this.forward_fn = forward_fn
		this.backward_fn = backward_fn
	}
	
	apply(values){
		return this.forward_fn(values)
	}
	
	iapply(values){
		return this.backward_fn(values)
	}
}

// input is in range (0,1) output is also in range (0,1)
// (0,0) maps to (0,0)
// (1,1) maps to (1,1)
// Only the middle bits change
// e**0 = 1
// e**1 = e
// e**2 = e*e
// log(0) = -inf
// log(1) = 0
// log(e) = 1

// z = e**b
// ln(z) = ln(e**b) = b
// z**a = (e**b)**(a) = (e**ab)
// logz(z**a) = a
// ln(e**ab) = ab
// ln(e**ab)/ln(z) = a
// ln(x)/ln(z) = logz(x)

let alpha = 1
let ln_alpha = Math.log(alpha)
let exp_alpha = Math.exp(alpha)

let base = 1000
let log_transform_y = new NonlinearTransform(
	(a)=>{
		//console.log("log_transform_y::forward_fn", a, Math.log10(a[1]), Math.log10(90*a[1]+10))
		let r = V.copy(a)
		r[1] = Math.log((base-1)*base*r[1]+base)/Math.log(base) - 1
		//r[1] = Math.log((Math.E-1)*Math.E*r[1] + 1)

		return r
	}, 
	(a)=>{
		//console.log("log_transform_y::backward_fn", a, Math.pow(10, a[1]*10))
		let r = V.copy(a)
		r[1] = (Math.pow(base, r[1]) - 1)/(base-1)
		//r[1] = (Math.exp(r[1]) - 1)/(Math.E -1)

		return r
	}
)
let identity_transform = new NonlinearTransform((x)=>x, (x)=>x)

class Axis{
	static n_instances = 0
	
	constructor({
		index,
		ndim,
		fromData_transform,
		toRoot_transform,
		containing_rect,
		pos_in_rect = 1,
		name = null,
		nonlinear_transform = null, // e.g. logarithm
		
	}={}){
		Axis.n_instances++
		this.name = (name===null) ? `axis_${Axis.n_instances}` : name
		this.index = index
		this.ndim = ndim
		this.fromData_transform = fromData_transform
		this.toRoot_transform = toRoot_transform
		this.containing_rect = containing_rect
		this.pos_in_rect = pos_in_rect
		this.nonlinear_transform = nonlinear_transform
		O.assert_has_attributes(this, "name", "index", "ndim", "fromData_transform", "toRoot_transform", "containing_rect", "pos_in_rect")
		
		this.label_anchor_pos = V.scalar_prod(V.ones(this.ndim),0.5)
		this.label_rotation = this.index*270 // rotation angle of axis lable
		this.display_direction_factors = V.ones(this.ndim)
		this.display_direction_factors[1] = 0
		
		
		
		this.path = null
		this.label_pos = null
		this.tick_set = []
		this.tick_label_pos_set = []
		this.tick_label_set = []
		this.tick_label_anchor_pos = null
		
		
		this.do_redraw_set = new Set() // to be filled with functions that redraw things when a redraw is required
		
		this.svg = new SvgContainer(Svg.group(`group-${this.name}`, null, {}))
		
		this.line_svg_holder = new SvgContainer(this.svg.add("group",`group-${this.name}-axis-line`, null, {}))
		this.label_svg_holder = new SvgContainer(this.svg.add("group",`group-${this.name}-axis-label`, null, {}))
		this.tick_set_svg_holder = new SvgContainer(this.svg.add("group",`group-${this.name}-axis-ticks`, null, {}))
		this.tick_label_set_svg_holder = new SvgContainer(this.svg.add("group",`group-${this.name}-axis-tick-labels`, null, {}))
		
	}
	
	addSvgTo(parent){
		parent.appendChild(this.svg.root)
	}
	
	updateContainingRect(rect){
		this.containing_rect = rect
		this.calcLine()
		
		this.redraw()
	}
	
	updateRootTransform(transform){
		this.toRoot_transform = transform
		
		this.draw() // everything needs to be redrawn if toRoot transform changes
	}
	
	updateFromDataTransform(transform){
		this.fromData_transform = transform
		
		this.calcTickLabels() // only the labels are updated if the fromData transform changes
		this.redraw()
	}
	
	redraw(){
		for(const redraw_callable of this.do_redraw_set){
			redraw_callable()
		}
		this.do_redraw_set.clear()
	}
	
	calc(){
		this.calcLine() // calcLine triggers the others to be calculated too
	}
	
	draw(){
		// Draw everything manually here
		this.drawLine()
		this.drawLabel()
		this.drawTicks()
		this.drawTickLabels()
		
		// as we have just drawn everything, remove everything from redraw_set
		this.do_redraw_set.clear()
	}
	
	calcLine(){
		//console.log("this.containing_rect", this.containing_rect)
		let a = V.const_of_size(this.ndim, this.pos_in_rect)
		a[this.index] = 0 // {0,POS} for x axis, {POS,0} for y-axis
		//console.log("a", a)
		let b = V.unit(this.index, this.ndim)
		//console.log("b", b)
		let start = V.add(V.prod(a,this.containing_rect.s), this.containing_rect.r)
		//console.log("start", start)
		this.path = V.from(
			...start,
			...V.add(start, b)
		)
		//console.log("this.path", this.path)
		this.do_redraw_set.add(this.drawLine.bind(this))
		
		// need to recalculate these as they depend on the line
		this.calcLabel()
		this.calcTicks()
	}
	
	drawLine(attrs={"stroke":"black"}){
		this.line_svg_holder.clear()
		
		this.line_svg_holder.add(
			"line",
			T.apply_block(this.toRoot_transform, this.path),
			attrs
		)
	}
	
	calcLabel(label_offset = 0.12){
		// if pos_in_rect > 0.5 should go in -ve direction, otherwise +ve direction
		let label_offset_dir = -1
		if(this.pos_in_rect <= 0.5){
			label_offset_dir = 1
		}
		let delta = V.scalar_prod(V.sub(V.ones(this.ndim), V.unit(this.index, this.ndim)), label_offset_dir*label_offset)
		this.label_pos = V.add(
			V.scalar_div(
				V.add(this.path.subarray(0,this.ndim), this.path.subarray(this.ndim, 2*this.ndim)),
				this.ndim
			),
			delta
		)
		this.do_redraw_set.add(this.drawLabel.bind(this))
	}
	
	drawLabel(){
		let pos = T.apply(this.toRoot_transform, this.label_pos)
		this.label_svg_holder.add(
			"text", 
			pos, 
			this.name, 
			{
				transform : `rotate(${this.label_rotation} ${pos})`,
			}, 
			this.label_anchor_pos
		)
	}
	
	calcTicks(n_ticks=7, tick_length=0.02){
		let tick_direction = -1
		if(this.pos_in_rect <= 0.5){
			tick_direction = 1
		}
	
		//console.log("this.path", this.path)
		let tick_displacement = V.scalar_prod(V.sub(V.ones(this.ndim),V.unit(this.index, this.ndim)), tick_direction*tick_length)
		//console.log("tick_displacement",tick_displacement)
		
		let starts = P.interpolatePointsAlong(this.path, n_ticks, this.ndim)
		//console.log("starts", starts)
		let ends = V.block_apply(V.add, starts, tick_displacement, this.ndim)
		//console.log("ends", ends)
		
		this.tick_set = []
		for(let i=0;i<n_ticks;i++){
			this.tick_set.push(
				V.from(...starts.subarray(i*this.ndim, (i+1)*this.ndim), ...ends.subarray(i*this.ndim, (i+1)*this.ndim))
			)
		}
		this.do_redraw_set.add(this.drawTicks.bind(this))
		
		// Need to recalculate these as they depend on the ticks
		this.calcTickLabelPositions()
		this.calcTickLabels()
	}
	
	drawTicks(
			attrs={
				stroke:"black",
				"stroke-width":0.2
			}
		){
		this.tick_set_svg_holder.clear() // remove all previous ticks
		
		for(const [i, tick] of this.tick_set.entries()){
			this.tick_set_svg_holder.add(
				"line",
				T.apply_block(this.toRoot_transform, tick),
				attrs
			)
		}
	}
	
	calcTickLabelPositions(offset=0.1){
		this.tick_label_pos_set = []
		this.tick_label_anchor_pos = V.scalar_prod(
			this.display_direction_factors,
			this.pos_in_rect
		)
		this.tick_label_anchor_pos[this.index] = 0.5
		
		//console.log("this.tick_label_anchor_pos", this.tick_label_anchor_pos)
		
		for(const [i, tick] of this.tick_set.entries()){
			this.tick_label_pos_set.push(
				V.add(
					tick.subarray(0,this.ndim),
					V.scalar_prod(
						V.sub(tick.subarray(this.ndim,2*this.ndim), tick.subarray(0,this.ndim)),
						1+offset
					)
				)
			)
		}
		this.do_redraw_set.add(this.drawTickLabels.bind(this))
	}
	
	calcTickLabels(){
		// gets the tick labels in data coordinates
		this.tick_label_set = []
		
		for(const [i, tick] of this.tick_set.entries()){
			//console.log(tick, this.nonlinear_transform.iapply(tick))
		
			this.tick_label_set.push(
				Svg.formatNumber(
					T.apply(
						this.fromData_transform, 
						(this.nonlinear_transform === null) ? tick : this.nonlinear_transform.iapply(tick)
					)[this.index]
				)
			)
		}
		this.do_redraw_set.add(this.drawTickLabels.bind(this))
	}
	
	drawTickLabels(){
		this.tick_label_set_svg_holder.clear() // remove previous tick labels
		
		for(let i=0; i<this.tick_label_set.length; i++){
			this.tick_label_set_svg_holder.add(
				"text",
				T.apply(this.toRoot_transform, this.tick_label_pos_set[i]),
				this.tick_label_set[i],
				{},
				this.tick_label_anchor_pos
			)
		}
	}
	
}

class AxesSet{
	// Knows about the data coordinate system
	// can transform from data coordinates to "dataArea" (unit) coordinates
	// Need to attach to a data area to be able to visualise axes
	
	static n_instances = 0
	
	constructor({
		name = null,
		//parent_frame = null,
		data_area = null, // place that data is draw into
		extent_in_data_coords = E.from(0,1,0,1),
		autoresize = [true, true], // autoresizing of each axis
		axis_positions = [0,0], // positions to draw axis compared to data area
		axis_names = [null,null], //names of component axis objects
		nonlinear_transform = identity_transform, // nonlinear transforms to apply to axes
	} = {}){
		//console.log("AxesSet::constructor name", name)
		
		AxesSet.n_instances++
		this.ndim = extent_in_data_coords.length/2
		this.name = (name===null) ? `axes_set_${AxesSet.n_instances}` : name
		this.axis_names = axis_names
		
		this.autoresize = autoresize
		this.axis_positions = axis_positions
		this.nonlinear_transform = nonlinear_transform
		
		this.registered_datasets = []
		this.original_extent = V.copy(extent_in_data_coords)
		this.setExtent(extent_in_data_coords)
		
		//console.log("this.toAxis_transform", toAxis_transform)
		
		this.data_area = null
		
		
		this.svg = new SvgContainer(Svg.group(`group-${this.name}`, null, {}))

		this.axis_list = []
		
		
		this.attachDataArea(data_area)
		
		//console.log("this.data_area", this.data_area)	
	}
	
	reset(){
		this.setExtent(V.copy(this.original_extent))
	}
	
	setExtent(extent_in_data_coords){
		//console.log("extent_in_data_coords2",extent_in_data_coords)
		assert(this.ndim == extent_in_data_coords.length/2, "New extext must have same number of dimensions as old extent")
		this.extent_in_data_coords = extent_in_data_coords
		this.fromData_transform = R.getTransformFromUnitCoordsTo(R.fromExtent(this.extent_in_data_coords))
		
		if (this.svg !== undefined){
			this.createSpatialAxes()
			this.draw()
		}
		
		if (this.data_area !== null){
			for(const dataset_name of this.registered_datasets){
				this.data_area.setDatasetTransform(dataset_name, this.fromData_transform, this.nonlinear_transform)
			}
		}
	}
	
	
	updateWhenExtentChanged(
			dimensions_updated, // a list of dimensions that have changed from previous value
		){ // -> bool : true if we needed to perform a redraw, false otherwise
		assert(this.ndim == dimensions_updated.length, "Cannot change number of dimensions of AxesSet after creation")

		//console.log("AxesSet::updateWhenExtentChanged()", this.extent_in_data_coords)
	
		this.fromData_transform = R.getTransformFromUnitCoordsTo(R.fromExtent(this.extent_in_data_coords))
		assert(!T.is_rank_deficient(this.fromData_transform), "this.fromData_transform must be of full rank")
		assert(!V.any_nan(this.fromData_transform), "this.fromData_transform not have any NaN entries")
		//console.log("New transform: ", this.fromData_transform)
		
		for( const[i, dim_updated_flag] of dimensions_updated.entries()){
			if(dim_updated_flag != 0){
				this.axis_list[i].updateFromDataTransform(this.fromData_transform)
			}
			else {
				// No need to bother with redraws if updated transform will not affect this axis
				this.axis_list[i].fromData_transform = this.fromData_transform
			}
		}
		
		if (this.data_area !== null){
			for(const dataset_name of this.registered_datasets){
				this.data_area.setDatasetTransform(dataset_name, this.fromData_transform, this.nonlinear_transform)
				this.data_area.clearData(dataset_name)
			}
		}
	}
	
	registerDataset(dataset_name){
		// Add a new dataset that this axes is responsible for representing
		let idx = this.registered_datasets.indexOf(dataset_name)
		if(idx < 0){
			this.registered_datasets.push(dataset_name)
		}
		this.data_area.registerDataset(dataset_name, this.fromData_transform)
	}
	
	deregisterDataset(dataset_name){
		// Remove a dataset that is axes is responsible for representing
		let idx = this.registered_datasets.indexOf(dataset_name)
		if(idx < 0){
			this.registered_datasets.splice(idx,1)
		}
		this.data_area.deregisterDataset(dataset_name)
	}
	
	drawDataset(dataset){
		//console.log("AxesSet::drawDataset()", dataset, this.name)
		// Update component axes if required, and draw dataset on the associated "data_area" of this axis
		//let resize = false
		let resize = V.of_size(this.ndim, Uint8Array) // uninitialised : {1,5,3,6,...} 
		let redraw_flag = false
		
		for(const data of dataset.getNewData(this.name)){
			//console.log("AxesSet::drawDataset() data entry in dataset", data)
			resize = V.scalar_prod_inplace(resize, 0) // {0,0,0,...}
			//console.log("AxesSet::drawDataset() resize", resize)
			for(const [i,ar_flag] of this.autoresize.entries()){
				//console.log("AxesSet::drawDataset() ar_flag", ar_flag)
				if(ar_flag){
					//console.log("AxesSet::drawDataset() this.extent_in_data_coords", this.extent_in_data_coords)
					//console.log("AxesSet::drawDataset() extent comps", this.extent_in_data_coords[i*2], this.extent_in_data_coords[i*2+1], data[i])
					if(!(this.extent_in_data_coords[i*2] <= data[i])){
						//console.log("AxesSet::drawDataset() data NOT less than lower bound of extent")
						resize[i] = 1
						this.extent_in_data_coords[i*2] = data[i]
					}
					if(!(data[i] <= this.extent_in_data_coords[i*2 + 1])){
						//console.log("AxesSet::drawDataset() data NOT greater than upper bound of extent")
						resize[i] = 1
						this.extent_in_data_coords[i*2+1] = data[i]
					}
					//console.log("AxesSet::drawDataset() resize", resize)
					
					// if our extent has no size, alter it slightly so it is bigger
					// this avoids problems with transforms later
					if (resize[i]==1 && !(this.extent_in_data_coords[i*2] != this.extent_in_data_coords[i*2+1])){
						//console.log("AxesSet::drawDataset() Trying to prevent rank deficient transform of axis", i)
						this.extent_in_data_coords[i*2] -= 50*Number.EPSILON
						this.extent_in_data_coords[i*2+1] += 50*Number.EPSILON
					}
				}
			}
			
			if (V.accumulate_sum(resize) > 0){
				redraw_flag = true
				this.updateWhenExtentChanged(resize)
			}
		
			//console.log("AxesSet::drawDataset() redraw flag", redraw_flag)
			
			if(!redraw_flag){
				this.data_area.drawData(dataset.name, data)
			}
			
		}
		return redraw_flag
	}
	
	attachDataArea(data_area){
		if(data_area === null){
			//console.log("Warning: Attempting to attach null data_area to axes_set")
			return
		}
		this.data_area = data_area
		this.createSpatialAxes()
	}
	
	createSpatialAxes(names = [], containing_rects = []){
		this.axis_list = []
		for(let i=0; i< this.data_area.dimensions.spatial; i++){
			this.createAxis(
				(names.length <= i) ? null : names[i],
				(containing_rects.length <= i) ? null : containing_rects[i],
			)
		}
	}
	
	getDefaultContainingRect(ax_idx){
		let axis_pos = (this.axis_positions[ax_idx] === null) ? 0 : this.axis_positions[ax_idx]
		let v_el = V.unit(ax_idx, this.data_area.dimensions.spatial)
		let v_not_el = V.sub(V.ones(this.data_area.dimensions.spatial), v_el)
		let a = 0.1
		
		let scale = V.add(
			v_el,
			V.scalar_prod(v_not_el, a)
		)
		
		let pos = V.scalar_prod(v_not_el, axis_pos)
		
		if(axis_pos < 0.5){
			pos = V.add(pos, V.scalar_prod(v_not_el, -a))
		}
		else if (axis_pos >= 0.5){
			// do nothing
		}
		
		return new R(
			...pos,
			...scale
		)
		
	}
	
	createAxis(name=null, containing_rect = null){
		if(containing_rect===null){
			containing_rect = this.getDefaultContainingRect(this.axis_list.length)
		}
		//console.log("containing_rect", containing_rect)
		this.axis_list.push(new Axis({
			index : this.axis_list.length,
			ndim : this.data_area.dimensions.spatial,
			fromData_transform : this.fromData_transform, // transform to go from data to "dataArea" (unit) coords
			toRoot_transform : this.data_area.frame.toRoot_transform, // transform to go from "dataArea" (unit) coords to "root" (screen) coords
			containing_rect : containing_rect,
			name : (this.axis_names[this.axis_list.length] === null) ?  `axis-${this.axis_list.length}` : this.axis_names[this.axis_list.length],
			pos_in_rect : 1 - ((this.axis_positions[this.axis_list.length] === null) ? 0 : this.axis_positions[this.axis_list.length]),
			nonlinear_transform : this.nonlinear_transform,
		}))
	}
	
	checkAxIdx(ax_idx){
		assert_not_null(ax_idx, "Index of axis to update must be a positive non-zero number less than the number of dimensions of the AxesSet")
		assert(Number.isInteger(ax_idx) && (ax_idx < this.ndim), "Index of axis to update must be a positive non-zero number less than the number of dimensions of the AxesSet")
	}
	
	updateAxisContainingRect(ax_idx, containing_rect = null){
		this.checkAxIdx(ax_idx)
		if (containing_rect !== null){
			this.axis_list[ax_idx].updateContainingRect(containing_rect)
		}
	}
	
	addSvgTo(parent){
		parent.appendChild(this.svg.root)
	}
	
	
	draw(){
		this.svg.clear()
		for(const [i,axis] of this.axis_list.entries()){
			if(this.axis_positions[i] !== null){
				axis.addSvgTo(this.svg.root)
				axis.calc()
				axis.draw()
			}
		}
	}
}


class DataArea{
	static n_instances = 0
		
	constructor({
		name = null,
		parent_frame = null,
		rect_in_parent = new R(0,0,1,1),
		dimensions = {
			spatial: 2,
		},
	} = {}){
		//console.log("name", name)
		
		DataArea.n_instances++
		this.name = (name===null) ? `data_area_${PlotArea.n_instances}` : name
		this.rect_in_parent = rect_in_parent
		this.frame = ReferenceFrame.fromRect(parent_frame, this.rect_in_parent)
		this.dimensions = dimensions
		
		//console.log("this.frame.toRoot_transform", this.frame.toRoot_transform)
		
		this.svg = new SvgContainer(Svg.group(`group-${this.name}`, null, {}))		
		
		this.next_dataset_index = 0 // dataset index counter
		this.dataset_transforms = new Map() // transform from data coords to DataArea coords
		this.dataset_nonlinear_transforms = new Map() // nonlinear transforms from DataArea coords to nonlinear DataArea coords 
		
		this.dataset_plot_type_artists = new Map() // artist for each dataset
		this.dataset_plot_type_artist_svg_nodes = new Map()
		this.dataset_svg_holders = new Map() // SVG holders for drawings of datasets

		
	}
	
	setDatasetPlotTypeArtist(dataset_name, plot_type_artist){
		assert_not_null(dataset_name)
		assert_not_null(plot_type_artist)
		
		let da = this.dataset_plot_type_artists.get(dataset_name)
		if(da !== undefined){
			// must remove plot_type_artist SVG from data area
			da.removeSvgFrom(this.svg.root)
		}
		this.dataset_plot_type_artists.set(dataset_name, plot_type_artist)
		plot_type_artist.addSvgTo(this.getDatasetHolder(dataset_name).root)
	}
	
	getDatasetPlotTypeArtist(dataset_name, default_plot_type_artist = null){
		let da = this.dataset_plot_type_artists.get(dataset_name)
		if(da === undefined){
			this.setDatasetPlotTypeArtist(
				dataset_name, 
				(default_plot_type_artist === null) ? new LinePlotArtist() : default_plot_type_artist
			)
			da = this.dataset_plot_type_artists.get(dataset_name)
			assert_not_null(da)
		}
		return da
	}
	
	setDatasetTransform(dataset_name, fromData_transform, nonlinear_transform = identity_transform){
		//console.log("fromData_transform", fromData_transform)
		/*
		this.dataset_transforms.set(
			dataset_name, 
			T.prod(
				this.frame.toRoot_transform,
				T.invert(fromData_transform)
			)
		)
		*/
		this.dataset_transforms.set(
			dataset_name,
			T.invert(fromData_transform)
		)
		//console.log("DataArea::setDatasetTransform()",nonlinear_transform)
		this.dataset_nonlinear_transforms.set(dataset_name, nonlinear_transform)
	}

	applyDatasetTransform(dataset_name, data){
		//console.log("DataArea::applyDatasetTransform()")
		//console.log(data)
		data = T.apply(this.dataset_transforms.get(dataset_name), data)
		//console.log(data)
		data = this.dataset_nonlinear_transforms.get(dataset_name).apply(data)
		//console.log(data)
		data = T.apply(this.frame.toRoot_transform, data)
		//console.log(data)
		return data
	}
	
	
	getDatasetTransform(dataset_name){
		return this.dataset_transforms.get(dataset_name)
	}
	
	newDatasetHolder(dataset_name){
		return new SvgContainer(
			this.svg.add(
				"group", 
				`dataset-${dataset_name}`,
				null, 
				{}
			)
		)
	}
	
	getDatasetHolder(dataset_name){
		let dataset_svg_holder = this.dataset_svg_holders.get(dataset_name)
		if (dataset_svg_holder === undefined){
			dataset_svg_holder = this.newDatasetHolder(dataset_name)
			this.dataset_svg_holders.set(dataset_name,dataset_svg_holder)
		}
		return dataset_svg_holder
	}
	
	registerDataset(dataset_name, fromData_transform){
		this.setDatasetTransform(dataset_name, fromData_transform)
		this.getDatasetPlotTypeArtist(dataset_name)
		this.getDatasetHolder(dataset_name)
	}
	
	deregisterDataset(dataset_name){
		this.dataset_transforms.delete(dataset_name)
		this.dataset_plot_type_artists.delete(dataset_name)
		this.dataset_svg_holders.delete(dataset_name)
	}
	
	clear(){
		for(const [dataset_name, plot_type_artist] of this.dataset_plot_type_artists.entries()){
			plot_type_artist.clear()
		}
	}
	
	clearData(dataset_name){
		let da = this.getDatasetPlotTypeArtist(dataset_name)
		if (da !== undefined){
			da.clear()
		}
	}
	
	drawData(dataset_name, data, default_plot_type_artist = null){
		//console.log("DataArea::drawData", dataset_name, data, default_plot_type_artist, this.getDatasetPlotTypeArtist(dataset_name, default_plot_type_artist))
		//this.getDatasetPlotTypeArtist(dataset_name, default_plot_type_artist).drawData(
		//	this.applyDatasetTransform(dataset_name, data)
		//)
		
		//console.log(this.dataset_transforms.get(dataset_name))
		
		//console.log(T.apply(
		//	this.dataset_transforms.get(dataset_name), 
		//	data
		//))
		
		//console.log(
		//	this.dataset_nonlinear_transforms.get(dataset_name).apply(
		//		T.apply(
		//			this.dataset_transforms.get(dataset_name), 
		//			data
		//		)
		//	)
		//)
		
		this.getDatasetPlotTypeArtist(dataset_name, default_plot_type_artist).drawDataWithTransform(
			this.dataset_nonlinear_transforms.get(dataset_name).apply(
				T.apply(
					this.dataset_transforms.get(dataset_name), 
					data
				)
			),
			this.frame.toRoot_transform
		)
		
	}
	
	addSvgTo(parent){
		parent.appendChild(this.svg.root)
	}
	
	draw(){
		return
		this.svg.add(
			"rect",
			T.apply_block(this.frame.toRoot_transform, E.from(0,0,1,1)),
			{stroke:"blue"}
		)
	}
	
	
	
}


class PlotArea{
	static n_instances = 0
	
	constructor({
		name = null,
		parent_frame = null,
		rect_in_parent = new R(0,0,1,1)
	} = {}){
		//console.log("name", name)
		
		PlotArea.n_instances++
		this.name = (name===null) ? `plot_area_${PlotArea.n_instances}` : name
		this.rect_in_parent = rect_in_parent
		this.frame = ReferenceFrame.fromRect(parent_frame, this.rect_in_parent)
		
		//console.log("this.frame.toRoot_transform", this.frame.toRoot_transform)
		
		this.current_axes = null
		this.current_dataset = null
		this.data_areas = new Map()
		this.axes = new Map()
		this.datasets = new Map()
		this.axes_for_dataset = new Map() // associate a dataset with one or more axes
		
		
		this.svg = new SvgContainer(Svg.group(`group-${this.name}`, null, {}))
		
		
	}
	
	clear(){
		for(const dataset of this.datasets.values()){
			dataset.clear()
		}
		for(const data_area of this.data_areas.values()){
			data_area.clear()
		}
		for(const axes of this.axes.values()){
			axes.reset()
		}
	}
	
	setDatasetPlotTypeArtist(dataset_name, plot_type_artist){
		for(const axes_name of this.axes_for_dataset.get(dataset_name)){
			//console.log("PlotArea::setDatasetPlotTypeArtist()", axes_name)
			this.axes.get(axes_name).data_area.setDatasetPlotTypeArtist(dataset_name, plot_type_artist)
		}
	}
	
	setCurrentAxes(axes_name){
		// Sets axes `axes_name` as the "current axes" i.e. the default unless a different axes is specified
		assert_not_null(axes_name)
		this.current_axes = axes_name
	}
	
	setCurrentDataset(dataset_name){
		// Sets dataset `dataset_name` as the "current_dataset" i.e. the default if a different dataset is not specified
		assert_not_null(dataset_name)
		this.current_dataset = dataset_name
	}
	
	setDataAsDataset(dataset_name, ...data){
		this.setAsDataset(data, dataset_name)
	}
	
	setAsDataset(data, dataset_name = null){
		//console.log("PlotArea::setAsDataset()", data, dataset_name)
		// Sets dataset `dataset_name` to only contain `data`
		if(dataset_name === null){
			dataset_name = this.current_dataset
		}
		let axes = null
		let dataset = this.datasets.get(dataset_name)
		let needs_redraw = false
		
		dataset.set(data)
		// send a message to axes for dataset telling it to draw the new point
		for(const axes_name of this.axes_for_dataset.get(dataset_name)){
			// if axes.drawDataset(...) returns true, we need to redra
			axes = this.axes.get(axes_name)
			needs_redraw = axes.drawDataset(dataset)
			//console.log("PlotArea::setAsDataset ", axes, needs_redraw)
			
			if(needs_redraw){				
				for(const redraw_dataset_name of axes.registered_datasets){
					//console.log("PlotArea::setAsDataset ", redraw_dataset_name)
					dataset = this.datasets.get(redraw_dataset_name)
					dataset.resetIterator(axes.name)
					axes.drawDataset(dataset)
				}
			}
			
		}
	}
	
	add_data_point(...data){
		this.appendToDataset(data)
	}
	
	addDataToDataset(dataset_name, ...data){
		this.appendToDataset(data, dataset_name)
	}
	
	appendToDataset(data, dataset_name = null){
		let axes = null
		let dataset = null
		let needs_redraw = false
		
		if(dataset_name === null){
			dataset_name = this.current_dataset
		}
		this.datasets.get(dataset_name).push(data)
		
		
		
		// send a message to axes for dataset telling it to draw the new point
		for(const axes_name of this.axes_for_dataset.get(dataset_name)){
			// if axes.drawDataset(...) returns true, we need to redra
			axes = this.axes.get(axes_name)
			dataset = this.datasets.get(dataset_name)
			needs_redraw = axes.drawDataset(dataset)
			
			//console.log("PlotArea::appendToDataset ", dataset_name, needs_redraw)
			
			if(needs_redraw){
				for(const redraw_dataset_name of axes.registered_datasets){
					//console.log("PlotArea::appendToDataset ", redraw_dataset_name)
					dataset = this.datasets.get(redraw_dataset_name)
					dataset.resetIterator(axes.name)
					axes.drawDataset(dataset)
				}
			}
			
		}
	}
	
	addSvgTo(parent){
		parent.appendChild(this.svg.root)
	}
	
	draw(){
		this.svg.add(
			"text",
			T.apply(this.frame.toRoot_transform, V.from(0.5,1)),
			this.name,
			{},
			V.from(0.5,0)
		)
		return
		this.svg.add(
			"rect",
			T.apply_block(this.frame.toRoot_transform, E.from(0,0,1,1)),
			{stroke:"green"}
		)
	}
	
	addDatasetToAxes(axes_name, dataset){
		this.datasets.set(dataset.name, dataset)
		let afd = this.axes_for_dataset.get(dataset.name)
		if(afd === undefined){
			afd = []
		}
		afd.push(axes_name)
		this.axes_for_dataset.set(dataset.name, afd)
		
		let axes = this.axes.get(axes_name)
		assert(axes !== undefined, `Attempting to add a dataset to axes name "${axes_name}" that does not exist. Axes names that exist are: ${Array.from(this.axes.keys())}`)
		
		axes.registerDataset(dataset.name)
		this.current_dataset = dataset.name
		this.current_axes = axes
	}
	
	newDatasetForAxes(axes_name, dataset_name = null){
		let ds = new Dataset(
			(dataset_name === null) ? `dataset-${this.datasets.size}` : dataset_name
		)
		this.addDatasetToAxes(axes_name, ds)
		return ds.name
	}
	
	appendDataArea(name, data_area){
		this.data_areas.set(name, data_area)
		data_area.addSvgTo(this.svg.root)
		this.frame.addChild(data_area.frame)
		data_area.draw()
	}
	
	appendDataAreaFromRect(name, d_rect_in_p){
		this.appendDataArea(
			name, 
			new DataArea({
				name : name,
				parent_frame : this.frame,
				rect_in_parent : d_rect_in_p
			})
		)
	}
	
	appendAxes(name, axes){
		this.axes.set(name, axes)
		axes.addSvgTo(this.svg.root)
		this.frame.addChild(axes.frame)
		axes.draw()
		this.current_axes = axes
	}
	
	appendAxesFromExtent(name, extent, axis_attrs = {}, data_area = null){
		//console.log(name, extent, data_area)
		if(data_area === null){
			if (this.data_areas.size > 0){
				data_area = this.data_areas.values().next().value
			}
		}
		
		// If one member of an extent is NaN
		// we should enable autoresizing in that direction
		let autoresize = new Array(
			(extent.length/2>>0) // integer division
		)
		for(let i=0; i<autoresize.length; ++i){
			if(Number.isNaN(extent[2*i]) || Number.isNaN[2*i+1]){
				autoresize[i] = true
			} else {
				autoresize[i] = false
			}
		}
				
		this.appendAxes(
			name,
			new AxesSet(
				O.insertIfNotPresent( // NOTE: earlier arguments take priority over later arguments
				{
					name : name,
					parent_frame : this.frame,
					extent_in_data_coords : extent,
					data_area : data_area,
				}, 
				axis_attrs,
				{
					autoresize: autoresize
				}
				)
			)
		)
	}
}

class Figure{

	static get_font_size(scale, units){
		switch (units){
			case "rem":
				return `${scale/20}rem`
			case "em":
				return `${scale/20}em`
			default:
				return `${scale/20}${units}`
		}
	}

	constructor({
		container,
		shape = V.from(6,4),
		units = "cm",
		scale = null,
		title = null,
		caption = null,
	} = {}){
		
		if (scale === null){
			// Assign scale so the largest value in "shape" (after multiplication by scale) is 100
			scale = 100/Math.abs(V.max(shape))
		}
		
		this.container = container
		this.set_title(title);
		this.set_caption(caption);
		// Unit coords go from zero to one in each dimension
		this.aspect_ratio = shape[0]/shape[1] // dx/dy
		this.f_rect_in_s = new R(0,scale*shape[1],scale*shape[0],-scale*shape[1])
		
		this.frame = ReferenceFrame.fromRect(null, this.f_rect_in_s)
		//console.log("this.frame.toRoot_transform", this.frame.toRoot_transform)
		
		this.plot_areas = new Map()
		
		this.svg = new SvgContainer(
			Svg.svg(
				shape, 
				units, 
				scale, 
				{
					"style":`font-size:${Figure.get_font_size(scale,units)};`, // |min({dx,dy}/{200,100})|
				}
			)
		)
		
		
		this.div_node = this.container.appendChild(Html.createElement("div", {class:["item","figure"]}))
		this.svg_node = this.div_node.appendChild(this.svg.root)
		
		this.draw()
	}
	
	set_title(title = null){
		this.title = title;
		
		if (this.title_node !== undefined){
			this.container.removeChild(this.title_node)
			delete this.title_node
		}
		if (this.title !== null){
			this.title_node = Html.createElement("p", {class:"title", textContent:this.title})
			this.div_node.insertBefore(this.title_node, this.svg_node)
		}
	}
	
	set_caption(caption = null){
		this.caption = caption;
		
		if (this.caption_node !== undefined){
			this.container.removeChild(this.caption_node)
			delete this.caption_node
		}
		if (this.caption !== null){
			this.caption_node = Html.createElement("div", {class:"caption"})
			this.div_node.insertBefore(this.caption_node, this.svg_node.nextSibling)
		
			if (Array.isArray(this.caption)){
				for(const c of this.caption){
					this.caption_node.appendChild(Html.createElement("p", {innerHTML:c}))
				}
			} else {
				this.caption_node.appendChild(Html.createElement("p", {innerHTML:this.caption}))
			}
		}
	}
	
	draw(){
		return
		this.svg.add(
			"rect",
			T.apply_block(this.frame.toRoot_transform, E.from(0,0,1,1)),
			{stroke:"red"}
		)
	}
	
	appendPlotArea(name, plot_area){
		this.plot_areas.set(name, plot_area)
		plot_area.addSvgTo(this.svg.root)
		this.frame.addChild(plot_area.frame)
		plot_area.draw()
	}
	
	appendPlotAreaFromRect(name, p_rect_in_f){
		this.appendPlotArea(
			name, 
			new PlotArea({
				name : name,
				parent_frame : this.frame,
				rect_in_parent : p_rect_in_f
			})
		)
	}
	
	
}

