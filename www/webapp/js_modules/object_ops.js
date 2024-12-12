"use strict"


function assert(condition, message='assertion failed'){
	if (condition === undefined){
		console.trace()
		throw Error('assert condition is undefined')
	}
	if (message === undefined){
		message = "assert condition failed"
	}
	if (!condition){
		console.trace(message)
		throw Error(message)
	}
}

function zip_arrays(...args){
	return args[0].map((v,i)=>{
		const a = new Array()
		for (const x of args){
			a.push(x[i])
		}
		return a
	})
}

class O{

	static isObject(obj){
		if((obj !== null) && (typeof(obj) === "object")){
			return true
		}
		return false
	}
	
	static *getClassesOf(obj){
		if(!O.isObject(obj)){
			throw Error("Cannot get classes of a non-object type")
		}
		let a = obj
		while(a.constructor != Object){
			a = Object.getPrototypeOf(a)
			yield a.constructor
		}
	}
	
	static *getBasesOf(obj){
		let klasses = Array.from(O.getClassesOf(obj))
		for(const klass of klasses.reverse()){
			yield klass
		}
	}

	static getStaticAttrOf(obj, attr){
		let temp = undefined
		let value = {}
		for (const klass of O.getBasesOf(obj)){
			temp = klass[attr]
			
			// Only do something if we have the attribue on the class
			if(temp !== undefined){
				if(O.isObject(temp)){
					Object.assign(value, temp)
				} else {
					value = temp
				}
			}
		}
		return value
	}

	static insertIfNotPresent(target, ...sources){
		for (const source of sources){
			for(const [key,value] of Object.entries(source)){
				if(!Object.hasOwn(target, key)){
					target[key] = value
				}
			}
		}
		return target
	}

	static assign(target, ...sources){
		for (const source of sources){
			for(const k of Object.keys(source)){
				target[k] = source[k]
			}
		}
		return target
	}

	static copy({source, exclude_keys=null, include_keys=null}={}){
		const obj = new Object()
		for(const k of Object.keys(source)){
			if ((include_keys === null) || (include_keys.includes(k))){
				if ((exclude_keys === null) || (!exclude_keys.includes(k))){
					obj[k] = source[k]
				}
			}
		}
		return obj
	}
	
	static assert_has_attribute(obj, attr_name){
		if ((!(attr_name in obj)) || (obj[attr_name] === undefined)){
			console.trace()
			throw Error(`Object ${obj} has no attribute "${attr_name}"`)
		}
	}
	
	static assert_has_attributes(obj, ...attr_names){
		for(const attr_name of attr_names){
			//console.log(`obj[${attr_name}]`, obj[attr_name])
			//console.log(`${attr_name} in obj`, attr_name in obj, `obj[${attr_name}] === undefined`, obj[attr_name] === undefined)
			if ((!(attr_name in obj)) || (obj[attr_name] === undefined)){
				console.trace()
				throw Error(`Object ${obj} has no attribute "${attr_name}"`)
			}
		}
	}




}