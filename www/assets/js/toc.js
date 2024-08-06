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


	static text_to_anchor(text){
		return text.toLowerCase()
			.replace(' ','-')
			.replace('/', '-back-slash-')
	}

	static from_headings_in(container){
		const elements = container.querySelectorAll("h1, h2, h3, h4, h5, h6")
		return new TableOfContents(
			container.ownerDocument,
			Pair.listFromIterable(elements, (e)=>{return new Pair(parseInt(e.nodeName[1]), e)})
		)
	}

	constructor(document, toc_level_to_element_pairs, list_type='ul', item_type='li'){
		this.doc = document
		this.list_type = list_type
		this.item_type = item_type
		
		this.toc_level = 0
		this.toc_level_stack = [this.doc.createElement('div')]
		this.current_level_element.setAttribute('class', 'table-of-contents')
		
		
		for (const pair of toc_level_to_element_pairs){
			console.log(pair[0], pair[1])
			this.process(pair[0], pair[1])
		}
	
	}
	
	as_child_of(html_parent_element){
		for(const child of this.toc_level_stack[0].childNodes){
			html_parent_element.appendChild(child) // child.cloneNode() if we get complaints about node being in two places at once.
		}
	}
	
	get current_level_element(){
		return this.toc_level_stack.length==0 ? null : this.toc_level_stack.at(-1)
	}
	
	process(level, element){
		while(this.toc_level < level){
			this.toc_level += 1
			this.push()
		}
		while (this.toc_level > level){
			this.toc_level -= 1
			this.pop()
		}
		this.add_entry_for(element)
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
		const text = element.textContent
		
		var anchor = element.getAttribute('id')
		if (anchor === null){
			anchor = TableOfContents.text_to_anchor(text)
			element.setAttribute('id', anchor)
		}
		
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
