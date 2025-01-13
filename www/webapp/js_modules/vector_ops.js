"use strict"

class V{

	static zero = Float64Array.from([0,0])
	static one =  Float64Array.from([1,1])
	
	static toBlockArray(v, ndim=2){
		let n_ndim_blocks = (v.length / ndim >>0)
		if(n_ndim_blocks*ndim < v.length){
			throw Error(`Must have v.length ${v.length} divisible by ndim ${ndim} if we want to manipulate v into a block array.`)
		}
		
		let a = new Array(n_ndim_blocks)
		
		for(let i=0; i<n_ndim_blocks;i++){
			a[i] = v.subarray(ndim*i, ndim*(i+1))
		}
		return a
	}
	
	static any_nan(v){
		for(const item of v){
			if (isNaN(item)){
				return true
			}
		}
		return false
	}
	
	static zeros(n=2,type=Float64Array){
		return V.const_of_size(n, 0, type)
	}
	static ones(n=2, type=Float64Array){
		return V.const_of_size(n, 1, type)
	}
	
	static unit(i,n=2, type=Float64Array){
		let v = new type(n)
		for (let j=0;j<n;j++){
			if(j==i){
				v[j] = 1
			} else {
				v[j] = 0
			}
		}
		return v
	}

	static of_size(n, type=Float64Array){
		return new type(n)
	}

	static from(...args){
		return Float64Array.from(args)
	}
	
	static from_matrix(m, type=Float64Array){
		return type.from(m.elements)
	}

	static select_by_idxs(a, ...idxs){
		let v1 = V.of_size(idxs.length)
		for(let i=0;i<idxs.length;i++){
			v1[i] = a[idxs[i]]
		}
		return v1
	}

	static typed_from(type, ...args){
		// type : Float64Array | Float32Array | Int32Array | Int16Array | Uint32Array | Uint16Array | Uint8Array | ...
		return type.from(args)
	}
	
	static const_of_size(n,c,type=Float64Array){
		let v = new type(n)
		for(let i=0; i<n;i++){
			v[i] = c
		}
		return v
	}

	static arange(start, stop, step=1, type=Float64Array){
		const n = Math.ceil((stop-start)/step)
		const r = new type(n)		
		for(let i=0; i<n; i++){
			r[i] = start + i*step
		}
		return r
	}

	static new_like(v1){
		//console.trace()
		return new v1.constructor(v1.length)
	}
	
	static copy(v1){
		const v2 = V.new_like(v1)
		V.assign(v2, v1)
		return v2
	}


	static check_length(v1,v2){
		//console.trace()
		if (v1.length != v2.length){
			throw new Error('Vectors must be of equal length to assign')
		}
	}

	static copyTo(v1, v2){
		V.check_length(v1,v2)
		for(let i=0; i<v1.length; i++){
			v1[i] = v2[i]
		}
		return v1
	}

	static assign(v1, v2){
		V.check_length(v1, v2)
		for(let i=0; i<v1.length; i++){
			v1[i] = v2[i]
		}
		return v1
	}

	static set(v1, ...args){
		V.check_length(v1, args)
		for(let i=0; i<v1.length; i++){
			v1[i] = args[i]
		}
		return v1
	}
	
	static block_apply(func, v1, v2, n=2){
		// apply func over blocks of the first argument
		// func(v1[0:n], v2)
		// func(v1[n:2n], v2)
		// func(v1[2n:3n], v2)
		// ...
		assert(Number.isInteger(v1.length/n), "When block applying, must have an integral number of blocks")
		let r = V.new_like(v1)
		for(let i=0; i<v1.length/n; ++i){
			V.copyTo(r.subarray(i*n,(i+1)*n), func(v1.subarray(i*n,(i+1)*n), v2))
		}
		return r
	}

	static mutate(v1, mutator, predicate=(i,v)=>{return true}){
		// Mutate the elements of v1 (i.e., change in place), using the `mutator` static, only when `predicate` is true.
		// mutator : callable = (index, original_value) -> new_value
		// 		Calculates the `new_value` of v1[index] from the `index` and `original_value` of v1[index]
		// predicate : callable = (index, original_value) -> bool
		// 		When this is true, v1[index] will be mutated using the `mutator` static
		
		
		for(let i=0; i < v1.length; i++){
			if (predicate(i,v1[i])){
				v1[i] = mutator(i,v1[i])
			}
		}
	}

	static component_min(v1,v2){
		// returns the minimum of each component between v1 and v2
		V.check_length(v1, v2)
		const r = V.new_like(v1)
		
		for(let i=0; i<v1.length; i++){
			r[i] = Math.min(v1[i], v2[i])
		}
		return r
	}
	
	static component_abs(v1){
		// returns the absolute value of each component in v1
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = Math.abs(v1[i])
		}
		return r
	}

	static intersection_rect(v1, v2, r1, s1){
		// v1 - start of a line
		// v2 - end of a line
		// r1 - bottom left of rectangle
		// s1 - size of rectangle
		return R.intersect(V.component_min(v1,v2), V.component_abs(V.sub(v1,v2)), r1, s1)
	}

	static add_many(...args){
		let r = V.new_like(args[0])
		for(const v of args){
			r = V.add_inplace(r, v)
		}
		return r
	}
	
	static add_inplace(v1, v2){
		V.check_length(v1, v2)
		
		for(let i=0; i<v1.length; i++){
			v1[i] += v2[i]
		}
		return v1
	}

	static add(v1, v2){
		V.check_length(v1, v2)
		
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] + v2[i]
		}
		return r
	}

	static sub(v1, v2){
		V.check_length(v1, v2)
		
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] - v2[i]
		}
		return r
	}
	
	static prod(v1, v2){
		V.check_length(v1, v2)
		
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] * v2[i]
		}
		return r
	}
	
	static div(v1, v2){
		V.check_length(v1, v2)
		
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] / v2[i]
		}
		return r
	}

	static dot(v1, v2){
		V.check_length(v1, v2)
		
		let r = 0
		for(let i=0; i<v1.length; i++){
			r += v1[i] * v2[i]
		}
		return r
	}
	
	static mag(v1){
		return Math.sqrt(V.dot(v1,v1))
	}
	
	static norm(v1){
		const r = V.new_like(v1)
		const a = Math.sqrt(V.dot(v1,v1))
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i]/a
		}
		return r
	}
	
	static abs(v1){
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = Math.abs(v1[i])
		}
		return r
	}
	
	static scalar_prod(v1, s){
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] * s
		}
		return r
	}
	
	static scalar_prod_inplace(v1, s){
		for(let i=0; i<v1.length; i++){
			v1[i] *= s
		}
		return v1
	}
	
	static scalar_add(v1, s){
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] + s
		}
		return r
	}
	
	static scalar_sub(v1, s){
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] - s
		}
		return r
	}
	
	static scalar_div(v1, s){
		const r = V.new_like(v1)
		for(let i=0; i<v1.length; i++){
			r[i] = v1[i] / s
		}
		return r
	}

	static accumulate_prod(v1){
		let r = 1
		for(const x of v1){
			r *= x
		}
		return r
	}
	
	static accumulate_sum(v1){
		let r = 0
		for(const x of v1){
			r += x
		}
		return r
	}

	static min(v1){
		let a = v1[0]
		for(const b of v1){
			if (b < a){
				a = b
			}
		}
		return a
	}
	
	static max(v1){
		let a = v1[0]
		for(const b of v1){
			if (b > a){
				a = b
			}
		}
		return a
	}

}



