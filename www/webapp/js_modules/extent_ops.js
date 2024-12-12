

// Extent
// Stores min and max of coordinate systems

// Requires:
// * vector_ops.js
// * rect_ops.js
// * obj_ops.js

class E{

	// Is an array-like that stores the min and max of each coord
	// it has the following layout:
	// {x0_min, x0_max, x1_min, x1_max, x2_min, x2_max,...}
	
	static from(...args){
		return V.from(...args)
	}
	
	static fromRect(rect){
		return V.from(
			rect.s[0] > 0 ? rect.r[0] : rect.s[0]+rect.r[0], 
			rect.s[1] > 0 ? rect.r[1] : rect.s[1]+rect.r[1], 
			rect.s[0] > 0 ? rect.r[0] + rect.s[0] : rect.r[0],
			rect.s[1] > 0 ? rect.r[1] + rect.s[1] : rect.r[1]
		)
	}
	
	static setExtent(extent, coord_idx, v){
		extent[coord_idx] = v[0]
		extent[coord_idx] = v[1]
		return extent
	}
	
	static getNDim(extent){
		return extent.length/2
	}
	
	static getScale(extent){
		let a = V.of_size(E.getNDim(extent))
		for(let i=0;i<extent.length;i+=2){
			a[i/2] = extent[i+1] - extent[i]
		}
		//console.log("getScale", a)
		return a
	}
	
	static getOffset(extent){
		let a = V.of_size(E.getNDim(extent))
		for(let i=0;i<extent.length;i+=2){
			//console.log(i, i/2, extent[i])
			a[i/2] = extent[i]
		}
		//console.log("getExtent", a)
		return a
	}
	
	static getTransformFromUnitCoordsTo(extent){
		// only work with 2-dimensional extents for now
		assert(E.getNDim(extent)==2, "extent must have only 2 dimensions")
		
		return T.from(...E.getScale(extent), ...E.getOffset(extent))
	}
	
	static *iterateOver(extent){
		for(let i=0; i<extent.length; i+=2){
			yield extent.subarray(i,i+2)
		}
	}
	
}