

class Svg{
	static createElement(tag, attributes={}){
		//console.trace()
		let element = document.createElementNS('http://www.w3.org/2000/svg',tag)
		return Svg.setAttrs(element, attributes)
	}
	
	static setAttrs(element, ...attr_objs){
		let value = null
		for (const attrs of attr_objs){
			for (const key of Object.keys(attrs)){
				value = attrs[key]
				if (key == "class"){
					if (value !==null ){
						element.classList.add(value) // classList is a set so no need to test for membership first
					}
				} else {
					element.setAttribute(key, value)
				}
			}
		}
		return element
	}
	
	static formatNumber(a, sig_figs=3, round_to=1E-12){
		if (round_to != 0){
			let diff = (a % round_to)
			a += (diff > round_to/2) ? diff : -diff
		}
		return a.toPrecision(sig_figs)
	}
	
	static svg(
		shape = V.from(4,6), //
		units = "cm", //
		scale = 1,
		attrs = {}
	){
	
		O.insertIfNotPresent(
			attrs, 
			{
				width:`${shape[0]}${units}`, // width of svg element
				height:`${shape[1]}${units}`, // height of svg element
				viewBox : `0 0 ${scale*shape[0]} ${scale*shape[1]}`, // viewbox of svg element
			}
		)
		return Svg.createElement('svg', attrs)
	}
	
	static n_group_instances
	static group(id = `group-${Svg.n_group_instances}`, transform = null, attrs={}){
		attrs.id = id
		if (transform !== null){
			attrs.transform = transform.toString()
		}
		return Svg.createElement('g', attrs)
	}
	
	static defs(attrs={}){
		return Svg.createElement('defs', attrs)
	}
	
	static marker(attrs={}){
		return Svg.createElement('marker', attrs)
	}
	
	static rect(extent, attrs={}){
		// return an svg rect
		// if height or width is -ve need to alter x and y coords
		let r = extent.subarray(0,2)
		let s = V.sub(extent.subarray(2,4), r)
		
		attrs.x = r[0] + (s[0] < 0 ? s[0] : 0)
		attrs.y = r[1] + (s[1] < 0 ? s[1] : 0)
		attrs.width = (s[0] < 0 ? - s[0] : s[0])
		attrs.height = (s[1] < 0 ? - s[1] : s[1])
		
		O.insertIfNotPresent(
			attrs,
			{
				"stroke":"black",
				"stroke-width":"0.5",
				"fill":"none",
				"stroke-opacity":0.3,
			}
		)

		return Svg.createElement("rect", attrs)
	}
	
	static path(path, attrs={}){
		//console.log("toSvgLinearPath", path)
		let svg_d_string = `M ${path[0]}, ${path[1]}`
		
		if(path.length > 2){
			svg_d_string += " L"
		}
		else {
			return svg_d_string
		}
		
		let i=0
		for(i=2; i<path.length; i+=2){
			svg_d_string += ` ${path[i]}, ${path[i+1]}`
		}
		attrs.d = svg_d_string
		
		O.insertIfNotPresent(
			attrs,
			{
				"stroke":"black",
				"stroke-width":0.5,
				"fill":"none",
				"stroke-opacity":0.3,
			}
		)
		
		return Svg.createElement("path", attrs)
	}
	
	static line(path, attrs={}){
		//console.log("path", path)
		attrs.x1 = path[0]
		attrs.y1 = path[1]
		attrs.x2 = path[2]
		attrs.y2 = path[3]
		
		O.insertIfNotPresent(
			attrs,
			{
				"stroke":"black",
				"stroke-width":"0.5",
				"fill":"none",
				"stroke-opacity":0.6,
			}
		)
		
		return Svg.createElement("line", attrs)
	}
	
	static polyline(path, attrs={}){
		attrs.points = path.toString()
		
		return Svg.createElement("polyline", attrs)
	}

	static circle(pos, r, attrs={}){
		attrs.r = r
		attrs.cx = pos[0]
		attrs.cy = pos[1]
		
		O.insertIfNotPresent(
			attrs,
			{
				"stroke":"black",
				"stroke-width":"0.5",
				"fill":"none",
				"stroke-opacity":0.3,
			}
		)
		
		return Svg.createElement("circle", attrs)
	}
	
	static square(pos, r, attrs={}){
		attrs.x = pos[0] - r/2
		attrs.y = pos[1] - r/2
		attrs.width = r
		attrs.height = r
	}
	
	static triangle(pos, r, attrs={}){
		triangle_path = V.from(
			pos[0]+ -1*r/2, pos[1]+ -0.5*r/2, 
			pos[0]+ 0*r/2 , pos[1]+ Math.sin(Math.PI/3)*r/2, 
			pos[0]+ 1*r/2 , pos[1]+ -0.5*r/2
		)
		return Svg.path(triangle_path, attrs)
	}
	
	static text(pos, content, attrs={}){
		// anchor_point : point on text bounding box that
		// should coincide with `pos`
		attrs.x = pos[0]
		attrs.y = pos[1]
		
		O.insertIfNotPresent(
			attrs,
			{
				style:"font-size : inherit;",
				"text-anchor":"middle",
				"font-family":"sans-serif",
				"stroke":"none",
				"stroke-width":"0.5",
				"fill":"black",
				"stroke-opacity":0.3,
				//"lengthAdjust" : "spacingAndGlyphs",
				//"textLength": "1cm",
			}
		)
		
		let text_el = Svg.createElement('text', attrs)
		text_el.textContent = content
		return text_el
	}
	
	static use(pos, id, transform, transform_origin, attrs={}){
		attrs.x = pos[0]
		attrs.y = pos[1]
		attrs.href = "#"+id
		if(transform !== null ){
			attrs.transform = transform
		}
		if(transform_origin !== null ){
			attrs["transform-origin"] = transform_origin
		}
		
		return Svg.createElement("use", attrs)
	}
}

class SvgContainer{
	constructor(
			svg, // svg to use as container
			//transform = T.identity, // transform to apply to all things
		){
		this.root = svg
		//this.transform = transform
	}
	
	add(tag, ...args){
		switch(tag){
			case "text":
				return this.addText(...args)
				break
			default:
				let element = Svg[tag](...args)
				this.root.appendChild(element)
				return element
		}
	}
	
	clear(){
		this.root.replaceChildren()
	}
	
	
	
	addText(pos, content, attrs={}, anchor_point=null, debug=false){
		// NOTE: anchor_point has its origin at the top left of the bounding box of the text
		let text_el = Svg.text(pos, content, attrs)
		this.root.appendChild(text_el)
		
		if (anchor_point === null){
			switch(attrs["text-anchor"]){
				case "start":
					anchor_point = V.from(0,1)
					break
				case "middle":
					anchor_point = V.from(0.5,1)
					break
				case "end":
					anchor_point = V.from(1,1)
					break
				default:
					anchor_point = V.from(0,1)
			}
		}
		
		//console.log("pos", pos)
		
		
		let bbox = text_el.getBBox()
		
		//console.log("bbox", bbox)
		
		let pos_anchor_in_bbox = V.prod(anchor_point, [bbox.width, bbox.height])
		//console.log("pos_anchor_in_bbox", pos_anchor_in_bbox)
		
		let pos_anchor = V.add([bbox.x, bbox.y], pos_anchor_in_bbox)
		//console.log("pos_anchor", pos_anchor)
		
		
		let diff = V.sub(pos_anchor, pos)
		//console.log("diff", diff)
		let new_pos = V.sub(pos, diff)
		//console.log("new_pos", new_pos)
		
		text_el.setAttribute("x", new_pos[0])
		text_el.setAttribute("y", new_pos[1])
		
		// Debug visuals
		if(debug){
			let debug_rect = new R(bbox.x-diff[0], bbox.y - diff[1], bbox.width, bbox.height)
			//console.log("debug_rect", debug_rect)
			//console.log("debug_rect_extent", E.fromRect(debug_rect))
			this.add("circle", pos_anchor, 0.2, {"stroke":"red"}) // original position of anchor debug visuals
			this.add("rect", E.from(bbox.x, bbox.y, bbox.x+bbox.width, bbox.y+bbox.height), {"stroke":"red"}) // original box debug visuals
			this.add("circle", pos, 0.2, {}) // new position of anchor debug visuals
			this.add("rect", E.fromRect(debug_rect)) // new box debug visuals
		}
		
		return text_el
	}
	
}