

class Html{
	static createElement(tag, attributes={}, ...attrs){
		let element = document.createElement(tag)
		return Html.setAttrs(element, attributes, ...attrs)
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
	
	static formatNumber(a, sig_figs=5, round_to=1E-12){
		if (round_to != 0){
			let diff = (a % round_to)
			a += (diff > round_to/2) ? diff : -diff
		}
		return a.toPrecision(sig_figs)
	}
}