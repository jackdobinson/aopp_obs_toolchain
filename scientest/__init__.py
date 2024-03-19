from types import SimpleNamespace

# test_output_dir is overwritten for each invocation of a test
# so it's ".path" attribute points to a specific folder
# created at runtime. However, for non scientest-managed invocation
# the ".path" attribute is set to "test_output"
test_output_dir = SimpleNamespace(path='test_output')