classdef Dropin < Layer
    properties
        p
    end
    
    methods
        function obj = Dropin(json)
            obj@Layer(FillDefault(json));
            obj.p = json.p;
            obj.Finalize();
        end
        
        function FP(obj)
            global plan;
            vars = obj.cpu.vars;
            out = zeros(size(vars.X));
            if (plan.training)
                input = plan.input;
                rand('seed', 100000 * input.repeat + input.step);
                ran = rand(size(vars.X));
                idx = logical(ran > obj.p);
                obj.cpu.vars.idx = idx;
                out(idx) = vars.X(idx);
                didx = logical(1 - idx);
                out(didx) = obj.cpu.vars.prev(didx);
            else
                out = vars.X;
            end
            obj.cpu.vars.out = out;
            obj.cpu.vars.prev = vars.X;
        end
        
        function BP(obj)                
            global plan
            data = obj.cpu.dvars.out;
            if (plan.training)
                dX = zeros(size(obj.cpu.vars.X));
                dX(obj.cpu.vars.idx) = data(obj.cpu.vars.idx);
            else
                assert(0);
            end      
            obj.cpu.dvars.X = dX;
        end
        
        function InitWeights(obj)
            global plan
            obj.AddParam('out', [prod(obj.dims(1:2)), obj.depth()], false);             
            obj.AddParam('prev', [plan.input.batch_size, obj.depth()], false);   
        end        
        
    end
end

function json = FillDefault(json)
json.type = 'Dropin';
json.one2one = true;
end
