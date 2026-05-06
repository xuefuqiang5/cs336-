import torch
import torch.nn as nn

class DeviceDetective:
    @staticmethod
    def check_model_params(model: nn.Module):
        """
        【静态检测】
        遍历模型所有参数和Buffer，检查是否有“掉队”在 CPU 的参数。
        """
        print(f"\n{'='*20} 开始检查模型参数位置 {'='*20}")
        device_map = {}
        has_issue = False
        
        # 1. 检查每一层的参数 (Weights/Biases)
        for name, param in model.named_parameters():
            device = param.device
            device_map.setdefault(device, []).append(f"Param: {name}")
        
        # 2. 检查 Buffer (例如 RoPE 的 cos/sin, BatchNorm 的 running_mean)
        for name, buf in model.named_buffers():
            device = buf.device
            device_map.setdefault(device, []).append(f"Buffer: {name}")

        # 3. 输出报告
        devices = list(device_map.keys())
        if len(devices) > 1:
            print(f"⚠️  警告：模型跨越了多个设备: {devices}")
            has_issue = True
            # 重点打印 CPU 的东西，因为通常这是 bug
            cpu_items = [x for x in devices if x.type == 'cpu']
            for d in cpu_items:
                print(f"❌ 发现以下模块滞留在 {d} (CPU):")
                for item in device_map[d][:10]: # 只打印前10个防止刷屏
                    print(f"   - {item}")
                if len(device_map[d]) > 10:
                    print(f"   ... 以及其他 {len(device_map[d])-10} 项")
        else:
            print(f"✅ 模型参数位置正常。统一在: {devices[0]}")
        
        print(f"{'='*60}\n")
        return has_issue

    @staticmethod
    def inspect_forward_flow(model: nn.Module):
        """
        【动态检测】
        给模型所有子模块注册 Hook，打印每一层输入数据的设备。
        一旦发现 输入数据设备 != 模块参数设备，就打印红色警告。
        """
        print(f"🕵️  已启动动态流检测 (Forward Hooks)...")
        
        def get_device(x):
            if isinstance(x, torch.Tensor):
                return x.device
            # 处理 tuple/list 输入的情况
            if isinstance(x, (tuple, list)):
                for item in x:
                    if isinstance(item, torch.Tensor):
                        return item.device
            return None

        def hook_fn(module, inputs, outputs):
            # 获取模块自身的设备（取第一个参数的设备）
            module_device = None
            try:
                module_device = next(module.parameters()).device
            except StopIteration:
                try:
                    module_device = next(module.buffers()).device
                except StopIteration:
                    pass # 既无参也无buffer的层（如ReLU），跳过对比
            
            # 获取输入数据的设备
            input_device = get_device(inputs)
            
            module_name = module.__class__.__name__

            # 判定逻辑
            if module_device and input_device and module_device != input_device:
                print(f"🔴 [CRITICAL ERROR] 模块 <{module_name}> 发生冲突！")
                print(f"   - 模块驻留于: {module_device}")
                print(f"   - 输入数据位于: {input_device}")
                print(f"   -> 这通常会导致 RuntimeError\n")
            elif input_device is not None and input_device.type == 'cpu':
                # 如果输入是 CPU，虽然可能不报错，但通常意味着性能瓶颈或忘记 .to(device)
                print(f"⚠️  [Warning] 模块 <{module_name}> 接收到了 CPU 数据。")

        # 注册 Hook
        hooks = []
        for name, module in model.named_modules():
            # 避免对顶层容器注册，只对叶子节点或主要层注册
            if len(list(module.children())) == 0 or isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)): 
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)
        
        return hooks