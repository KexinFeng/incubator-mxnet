class Intermediate:
    def __init__(self, name, data=None, grad_req='write'):
        self._name = name
        self._data = data
        self._grad_req = grad_req
    
    def __repr__(self):
        s = 'Intermediate name={name}'
        return s.format(name=self._name)

    def data(self):
        return self._data
    
    @property
    def name(self):
        return self._name
    
    @property
    def grad_req(self):
        return self._grad_req
    
    
