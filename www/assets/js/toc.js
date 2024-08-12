"use strict"

export class Pair{
	
	static listFromIterable(iterable, pair_creator){
		const a = []
		for(const item of iterable){
			a.push(pair_creator(item))
		}
		return a
	}
	
	constructor(a,b){
		this[0] = a
		this[1] = b
	}
}

export class TableOfContents{
	/*
	TODO: Ensure each ID is unique within a page. Possible solution: If an ID is not unique, prepend parent ids (delimited with colon) until it is unique. If still not unique, append an increasing integer.
	*/

	static text_to_anchor(text){
		return text
			.trim()
			.toLowerCase()
			.replaceAll(' ','-')
			.replaceAll('/', '-back-slash-')
	}

	static from_headings_in(container){
		const elements = container.querySelectorAll("h1, h2, h3, h4, h5, h6")
		return new TableOfContents(
			container.ownerDocument,
			Pair.listFromIterable(elements, (e)=>{return new Pair(parseInt(e.nodeName[1]), e)}) // list of (html_tag, element) pairs.
		)
	}

	constructor(document, toc_level_to_element_pairs, list_type='ol', item_type='li'){
		this.doc = document
		this.list_type = list_type
		this.item_type = item_type
		
		this.toc_level = 0
		this.toc_level_stack = [this.doc.createElement('div')]
		this.current_level_element.setAttribute('class', 'table-of-contents')
		this.current_level_element.setAttribute('id', 'contents-root')
		
		this.used_ids = []
		this.last_anchor_stack = []
		
		
		for (const pair of toc_level_to_element_pairs){
			//console.log(pair[0], pair[1])
			this.process(pair[0], pair[1])
		}
	
	}
	
	as_child_of(html_parent_element, element_to_delete=null, delete_if_no_toc=false){
		var n_toc_entries = 0
		for(const child of this.toc_level_stack[0].childNodes){
			html_parent_element.appendChild(child) // child.cloneNode() if we get complaints about node being in two places at once.
			n_toc_entries += 1
		}
		if ((n_toc_entries == 0) && (delete_if_no_toc)){
			if (element_to_delete === null){
				element_to_delete = html_parent_element
			}
			element_to_delete.remove()
		}
	}
	
	get current_level_element(){
		return this.toc_level_stack.length==0 ? null : this.toc_level_stack.at(-1)
	}
	
	process(level, element){
		while(this.toc_level < level){
			if (this.last_anchor_stack.at(-1) !==  null){
				this.current_level_element.setAttribute('id', 'contents-'+this.last_anchor_stack.at(-1))
			}
			this.toc_level += 1
			this.push()
			this.last_anchor_stack.push(null)
		}
		while (this.toc_level > level){
			this.toc_level -= 1
			this.pop()
			this.last_anchor_stack.pop()
		}
		this.add_entry_for(element)
		if (this.last_anchor_stack.at(-1) ===  null){
			this.last_anchor_stack.at(-1) = this.current_level_element.lastChild.getAttribute('id')
		}
	}
	
	new_list_container() {
		const list = this.doc.createElement(this.list_type)
		list.setAttribute('class', `level-${this.toc_level} empty`)
		this.current_level_element.appendChild(list)
		return list
	}
	
	new_list_item(anchor, text){
		const item = this.doc.createElement(this.item_type)
		const a = this.doc.createElement('a')
		a.setAttribute('href', `#${anchor}`)
		a.textContent = text
		item.appendChild(a)
		return item
	}
	
	
	add_entry_for(element){
		var text = element.textContent
		
		//var anchor = element.getAttribute('id')
		//if (anchor === null){
		var anchor = null;
		var cle = this.current_level_element
		var i=1
		var prefix = null
		var candidate
		if (anchor === null){
			anchor = TableOfContents.text_to_anchor(text)
			candidate = anchor
			console.log(cle)
			while (this.used_ids.some((x)=>x==candidate)){
				if (cle !== null){
					prefix = cle.getAttribute('id')
					cle = cle.parentNode
				} else {
					prefix = null
				}
				if (prefix !== null){
					if (prefix.startsWith('contents-')){
						prefix = prefix.slice(9)
					}
					anchor = prefix + '--' + anchor
					candidate = anchor
				} else {
					candidate = anchor + `-${i}`
					i += 1
				}
			}
		}
		anchor = candidate
		
		element.setAttribute('id', anchor)
		this.used_ids.push(anchor)
		console.log(`anchor=${anchor}`) // DEBUGGING
		
		this.current_level_element.appendChild(this.new_list_item(anchor, text))
		this.current_level_element.setAttribute('class',
			this.current_level_element.getAttribute('class').replace(' empty', '')
		)
	}
	
	push(){
		this.toc_level_stack.push(this.new_list_container())
	}
	
	pop(){
		this.toc_level_stack.pop()
	}
	
	

}
