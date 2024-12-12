

class MapOp{
	static insert_or_update_on_predicate(
			map, // Map() object instance
			key, // Key to be set or updated
			new_value, // Value to be assigned
			insert_predicate=(new_value)=>true, // If key does not exist already, only insert when this predicate is true
			update_predicate=(old_value, new_value)=>true // if key already present, only update when this predicate is true
		){
		const old_value = map.get(key)
		if (old_value===undefined){
			if (insert_predicate(new_value)){
				map.set(key,new_value)
				return true
			}
		} else if (update_predicate(old_value, new_value)){
			map.set(key,new_value)
			return true
		}
		return false
	}
	
}