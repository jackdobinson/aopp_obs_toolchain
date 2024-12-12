"use strict"

// Requires:
// * vector_ops.js

class P{
	// Path is a list of points in the format {x0_0, x1_0, x2_0, ..., xn_0, x0_1, ..., xn_1, ..., x0_m, ..., xn_m}
	// Where n is the number of dimensions, and m is the number of points the path
	static unit_path = V.from([0,0,1,1]) // x1,y1,x2,y2,...

	static *iterateOver(path, n=2){
		for(let i=0;i<path.length;i+=n){
			yield path.subarray(i,i+n)
		}
	}
	
	static at(path, i, n=2){
		let j = i<0 ? path.length/n - i : i
		return path.subarray(j*n, (j+1)*n)
	}
	
	static set(path, i, v, n=2){
		let j = i<0 ? path.length/n - i : i
		for(let k=0; k<n; k++){
			path[j*n+k] = v[k]
		}
		return path.subarray(j*n, (j+1)*n)
	}

	static interpolateBetween(p0, p1, n){
		// n - number of path segments, 1 greater than number of points
		//console.log("interpolate between", p0, p1, n)
		let path = V.of_size(2*(n+1))
		let dx = 1/n
		let i = 0;
		for(i=0; i<(2*(n+1)); i+=2){
			path[i] = p0[0] + (i/2)*dx*(p1[0] - p0[0])
			path[i+1] = p0[1] + (i/2)*dx*(p1[1] - p0[1])
		}
		return path
	}
	
	static interpolatePointsAlong(path, n, ndim=2){
		// given a path, interpolate n points along it (including end points)
		let points = V.of_size(n*ndim)
		// sum length of path
		let pathlen = 0
		let seglen = 0
		let i = 0
		for(const segment of P.iterateOver(path, 2*ndim)){
			seglen = 0
			for(i=0;i<ndim;++i){
				seglen += (segment[ndim+i]-segment[i])*(segment[ndim+i]-segment[i])
			}
			pathlen += Math.sqrt(seglen)
		}
		//console.log("pathlen", pathlen)
		
		// distance to emit points
		let k=0
		let j=0
		let delta = pathlen/(n-1)
		let frac_along_segment = 0

		pathlen = 0 // reset to zero
		// Loop over segments and emit points
		for(const segment of P.iterateOver(path, 2*ndim)){
			//console.log("segment", segment)
			seglen = 0
			for(i=0;i<ndim;++i){
				seglen += (segment[ndim+i]-segment[i])*(segment[ndim+i]-segment[i])
			}
			seglen = Math.sqrt(seglen)
			frac_along_segment = (j*delta - pathlen)/seglen
			//console.log("pathlen", pathlen)
			//console.log("seglen", seglen)
			
			
			while(frac_along_segment <= 1){
				//console.log("frac_along_segment", frac_along_segment)
				// this segment has a point we should emit
				
				//console.log(segment.subarray(ndim,2*ndim)) // end
				//console.log(segment.subarray(0,ndim)) // start
				//console.log(V.sub(segment.subarray(ndim,2*ndim),segment.subarray(0,ndim))) // vector along segment
				//console.log(V.scalar_prod(V.sub(segment.subarray(ndim,2*ndim),segment.subarray(0,ndim)), frac_along_segment)) // fraction along segment
				//console.log( V.add(segment.subarray(0,ndim), V.scalar_prod(V.sub(segment.subarray(ndim,2*ndim),segment.subarray(0,ndim)), frac_along_segment)))
				
				let a = V.add(segment.subarray(0,ndim), V.scalar_prod(V.sub(segment.subarray(ndim,2*ndim),segment.subarray(0,ndim)), frac_along_segment))
				for(k=0; k<ndim; k++){
					points[j*ndim + k] = a[k] 
				}
				
				
				j++
				frac_along_segment = (j*delta - pathlen)/seglen
			}
			pathlen += seglen
		}
		return points
	}
	
	static toSvgLinearPath(path){
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
		return svg_d_string
	}

	
	

}