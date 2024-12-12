"use strict"

// Requires:
// * vector_ops.js
// * transform_ops.js

class R{
	
	static fromPoints(p){
		return new R(p[0][0],p[0][1],p[1][0]-p[0][0],p[1][1]-p[0][1])
	}
	
	static fromExtent(e){
		// Extent is {x_min, x_max, y_min, y_max, ...}
		//return new R(e[0], e[1], e[2]-e[0], e[3]-e[1])
		return new R(e[0], e[2], e[1]-e[0], e[3]-e[2])
	}

	static fromUnitRectWithinRect(r, u){
		// u is a unit rect within r
		// u = (0,0,1,1) should return r
		return new R(...V.add(r.r, V.prod(u.r, r.s)), ...V.prod(r.s, u.s))
	}

	constructor(x,y,w,h){
		this.r = V.from(x,y)
		this.s = V.from(w,h)
		
		Object.seal(this)
	}
	
	asPoints(){
		return [V.from(this.r[0], this.r[1]), V.from(this.r[0]+this.s[0], this.r[1]+this.s[1])]
	}
	
	applyTransform(t){
		let b = new R(0,0,1,1)
		b.r = T.apply(t, this.r)
		b.s = T.scale(t, this.s)
		return b
	}
	
	applyTransformInplace(t){
		this.r = T.apply(t, this.r)
		this.s = T.scale(t, this.s)
		return this
	}
	
	asString(units=""){
		return `${this.r[0]}${units} ${this.r[1]}${units} ${this.s[0]}${units} ${this.s[1]}${units}`
	}
	
	asSvg(attrs){
		// return an svg rect
		// if height or width is -ve need to alter x and y coords
		let rect = document.createElementNS('http://www.w3.org/2000/svg',"rect")
		attrs.x = this.r[0] + (this.s[0] < 0 ? this.s[0] : 0)
		attrs.y = this.r[1] + (this.s[1] < 0 ? this.s[1] : 0)
		attrs.width = (this.s[0] < 0 ? -this.s[0] : this.s[0])
		attrs.height = (this.s[1] < 0 ? -this.s[1] : this.s[1])
		
		for (const k of Object.keys(attrs)){
			rect.setAttribute(k, attrs[k])
		}
		return rect
	}
	
	static getTransformFromUnitCoordsTo(a){
		// get a transform from unit coords (0,0,1,1) to rect coords
		return T.from(a.s[0], a.s[1], a.r[0],a.r[1])
	}
	
	static getTransformToUnitCoordsFrom(a){
		// get a transform to unit coords (0,0,1,1) from rect coords
		return T.invert(R.getTransformFromUnitCoordsTo(a))
	}

	static getTransformFromTo(a, b){
		// Get a transform from rect a coords to rect b coords
		// both a and b have to be in the same coord system
		let u2a_transform = R.getTransformFromUnitCoordsTo(a)
		let u2b_transform = R.getTransformFromUnitCoordsTo(b)
		return T.prod(u2b_transform, T.invert(u2a_transform)) // a2u then u2b gives a2b
	}

	static intersect(r1, s1, r2, s2){
		// r1, r2 are positions of rect corners
		// s1, s2 are width and height of rects.
		//
		// Tests if rect1 and rect2 intersect, returns intersection rect if they do, or null if they dont
		// rect1 : [r1x, r1y, s1w, s1h]
		// rect2 : [r2x, r2y, s2w, s2h]
		
		
		// If both corners of rect1 are on one side of rect 2, then they do not intersect
		if ( 
			((r1[0]+s1[0]) < r2[0])
			|| (r1[0] > (r2[0]+s2[0]))
			|| ((r1[1]+s1[1]) < r2[1])
			|| (r1[1] > (r2[1]+s2[1]))
		){
			return null
		}
		
		const r3 = new Float64Array(2)
		const s3 = new Float64Array(2)
		
		r3[0] = Math.max(r1[0],r2[0])
		r3[1] = Math.max(r1[1], r2[1])
		s3[0] = Math.min(r1[0]+s1[0], r2[0]+s2[0]) - r3[0]
		s3[1] = Math.min(r1[1]+s1[1], r2[1]+s2[1]) - r3[1]
		
		return [r3, s3]
	}
}