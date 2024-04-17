class GradCamHook:
    def __init__(self, module):
        self.module = module
        self.activations = None
        self.gradients = None
        
    def forward_hook(self, module, input, output):
        self.activations = output
        
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def register(self):
        self.handle_forward = self.module.register_forward_hook(self.forward_hook)
        self.handle_backward = self.module.register_backward_hook(self.backward_hook)
    def remove(self):
        self.handle_forward.remove()
        self.handle_backward.remove()
