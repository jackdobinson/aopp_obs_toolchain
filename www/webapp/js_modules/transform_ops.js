"use strict"

class T{

	static identity = Float64Array.from([1,0,0,1,0,0])

	static toSvgString(t){
		return "matrix(" + t.toString() + ")"
	}

	static ensure_full_rank(t){
		// If transform is not full rank, add a very small amount to the
		// scale elements to ensure the value is small but not completely zero
		t[0] += ((t[0]==0) ? 100*Number.EPSILON : 0)
		t[3] += ((t[3]==0) ? 100*Number.EPSILON : 0)
		return t
	}
	
	static is_rank_deficient(t){
		return (t[0] == 0 || t[3] == 0)
	}

	static from_full(a11, a21, a12, a22, a31, a32){
		const t = new Float64Array(6)
		t[0] = a11
		t[1] = a21
		t[2] = a12
		t[3] = a22
		t[4] = a31
		t[5] = a32
		return t
	}
	
	static from(sx, sy, tx, ty){
		const t = new Float64Array(6)
		t[0] = sx //(sx ==0) ? 100*Number.EPSILON : sx 
		t[1] = 0
		t[2] = 0
		t[3] = sy //(sy == 0) ? 100*Number.EPSILON : sy
		t[4] = tx
		t[5] = ty
		return t
	}

	static from_rotation(theta){
		const t = new Float64Array(6)
		const c = Math.cos(theta)
		const s = Math.sin(theta)
		t[0] = c
		t[1] = s
		t[2] = -s
		t[3] = c
		t[4] = 0
		t[5] = 0
		return t
	}

	static from_rotation_about_point(tx, ty, theta){
		const t = new Float64Array(6)
		const c = Math.cos(theta)
		const s = Math.sin(theta)
		t[0] = c
		t[1] = s
		t[2] = -s
		t[3] = c
		t[4] = -tx*c + ty*s + tx
		t[5] = -tx*s - ty*c + ty
		return t
	}

	static prod(t, u){
		// Matrix product of two transforms
		const r = new Float64Array(6)
		
		r[0] = t[0]*u[0] + t[2]*u[1]
		r[1] = t[1]*u[0] + t[3]*u[1]
		r[2] = t[0]*u[2] + t[2]*u[3]
		r[3] = t[1]*u[2] + t[3]*u[3]
		r[4] = t[0]*u[4] + t[2]*u[5] + t[4]
		r[5] = t[1]*u[4] + t[3]*u[5] + t[5]
		
		return r
	}
	
	static prod_many(...args){
		// Matrix product of two transforms
		
		let r = Float64Array.from(T.identity)
		
		args.reverse()
		for (let i=0; i<args.length; i++){
			r = T.prod(args[i], r) // multiply from right to left to keep "matrix-like" ordering
		}
		return r
	}

	static with_scale(t, sx, sy){
		r = Float64Array.from(t)
		T.scale_this(r, sx, sy)
		return r
	}

	static scale_this(t, sx, sy){
		t[0] *= sx
		t[3] *= sy
		t[4] *= sx
		t[5] *= sy
	}

	static transform_this(t, tx, ty){
		t[4] += tx
		t[5] += ty
	}

	static rotate_this(t, theta){
		const c = Math.cos(theta)
		const s = Math.sin(theta)
		
		t[0] = t[0]*c - t[1]*s
		t[1] = t[0]*s + t[1]*c
		t[2] = t[2]*c - t[3]*s
		t[3] = t[2]*s + t[3]*c
		t[4] = t[4]*c - t[5]*s
		t[5] = t[4]*s + t[5]*c
	}

	static invert(t){
		const r = new Float64Array(6)
		const a = t[1]*t[2] - t[0]*t[3]
		r[0] = -t[3]/a
		r[1] = t[1]/a
		r[2] = t[2]/a
		r[3] = -t[0]/a
		r[4] = (-t[2]*t[5] + t[3]*t[4])/a
		r[5] = (t[0]*t[5]-t[1]*t[4])/a
		return r
	}

	static iapply(t, p){
		const q = new Float64Array(2)
		const a = t[1]*t[2]-t[0]*t[3]
		q[0] = (-t[2]*t[5] - t[3]*p[0] + t[2]*p[1] + t[4]*t[3])/a //(p[0] - t[4])/t[0]
		q[1] = (t[0]*t[5] + t[1]*p[0] - t[0]*p[1] - t[4]*t[1])/a //(p[1] - t[5])/t[3]
		return q
	}
	
	static iapply_block(t, p){
		const q = new Float64Array(p.length)
		const a = t[1]*t[2]-t[0]*t[3]
		for (let i=0; i<p.length; i+=2){
			q[i] = (-t[2]*t[5] - t[3]*p[i] + t[2]*p[i+1] + t[4]*t[3])/a
			q[i+1] = (t[0]*t[5] + t[1]*p[i] - t[0]*p[i+1] - t[4]*t[1])/a
		}
		return q
	}

	static apply(t, q){
		const p = new Float64Array(2)
		p[0] = q[0]*t[0] + q[1]*t[2] + t[4]
		p[1] = q[0]*t[1] + q[1]*t[3] + t[5]
		return p
	}
	
	static apply_block(t, q){
		const p = new Float64Array(q.length)
		for (let i=0; i<q.length; i+=2){
			p[i] = q[i]*t[0] + q[i+1]*t[2] + t[4]
			p[i+1] = q[i]*t[1] + q[i+1]*t[3] + t[5]
		}
		return p
	}

	static iscale(t, s){
		const r = new Float64Array(2)
		const a = t[1]*t[2]-t[0]*t[3]
		r[0] = (-t[3]*s[0] + t[2]*s[1])/a
		r[1] = (t[1]*s[0] - t[0]*s[1])/a
		return r
	}

	static scale(t, s){
		const r = new Float64Array(2)
		r[0] = s[0]*t[0] + t[2]*s[1]
		r[1] = t[1]*s[0] + t[3]*s[1]
		return r
	}

	static scalar_scale_along_dim(t, i, x){
		// scale a value as if it were along a specific dimension
		if (i==0){
			return t[0]*x
		}
		if (i==1){
			return t[3]*x
		}
		throw new Error("Unrecognised dimension number for transform scaling a scalar along a dimension")
	}
}