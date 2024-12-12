"use strict"

// Requires:
// * vector_ops.js

class M{
	constructor(nrows, ncols, type=Float64Array){
		//console.log(nrows, ncols, type)
		this.nrows = nrows
		this.ncols = ncols
		this.elements = new type(nrows*ncols)
		
		Object.seal(this) // All attributes must be defined above this line
		/*
		shape n,m; store column major
		/ a11 a12 a13 ... a1m \
		| a21 ... ... ... a2m |
		| ... ... ... ... ... |
		\ an1 ... ... ... anm /
		*/
	}
	
	*[Symbol.iterator](){
		// stored column major order, therefore iterate over columns
		for (let c=0; c<this.ncols; c++){
			yield this.elements.subarray(c*this.nrows, (c+1)*this.nrows) // yields a vector for each column
		}
	}
	
	at(index){
		return this.elements[index]
	}
	
	set_at(index, value){
		this.elements[index] = value
	}
	
	rc_at(row,col){
		return this.elements[row + col*this.nrows]
	}
	
	set_rc_at(row, col, value){
		this.elements[row + col*this.nrows] = value
	}
	
	add_rc_at(row, col, value){
		this.elements[row + col*this.nrows] += value
	}
	
	static from(array, nrows, ncols){
		const a = new M(nrows, ncols)
		V.assign(a.elements, array)
		return a
	}
	
	static from_vector(v){
		const a = new M(v.length, 1)
		V.assign(a.elements, v)
		return a
	}
	
	static from_nested(args){
		const a = new M(args[0].length, args.length)
		for (let r=0; r<a.nrows; r++){
			for(let c=0; c<a.ncols; c++){
				a.set_rc_at(r,c, args[c][r])
			}
		}
		return a
	}
	
	static typed_from_nested(type, args){
		//console.log(args)
		const a = new M(args[0].length, args.length, type)
		for (let r=0; r<a.nrows; r++){
			for(let c=0; c<a.ncols; c++){
				a.set_rc_at(r,c, args[c][r])
			}
		}
		//console.log(a)
		return a
	}
	
	static transpose(a){
		const b = M(a.ncols, a.nrows, a.elements.constructor)
		for (let r=0; r<a.nrows; r++){
			for (let c=0; c<a.ncols; c++){
				b.set_rc_at(c,r, a.rc_at(r,c))
			}
		}
		return b
	}
	
	static rotation2d(radians){
		const ndim = 2
		const a = new M(ndim, ndim)
		a.set_rc_at(0,0,Math.cos(radians))
		a.set_rc_at(1,0,Math.sin(radians))
		a.set_rc_at(0,1,-Math.sin(radians))
		a.set_rc_at(1,1,Math.cos(radians))
		return a
	}
	
	static multiply_check_shapes(a, b){
		if( a.ncols == b.nrows){
			Error('Matrix multiplication requires equal rows and columns of first and second argument')
		}
	}
	
	static multiply(a, b){
		/*
		a11*b11 + a12*b21     a11*b12+a12*b22
		a21*b11 + a22*b21     a21*b12+a22*b22
		*/
		M.multiply_check_shapes(a,b)
		
		const r = new M(a.nrows, b.ncols)
		
		for(let i=0; i<a.nrows; i++){
			for (let j=0; j<b.ncols; j++){
				r.set_rc_at(i,j,0)
				for (let k=0; k<a.ncols; k++){
					r.add_rc_at(i,j, a.rc_at(i,k)*b.rc_at(k,j))
				}
			}
		}
		return r
	}
	
}