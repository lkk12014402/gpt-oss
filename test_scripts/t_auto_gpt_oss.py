import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights
import torch
model_path = "./gpt-oss-20b-int4-AutoRound"
ori_model_path = "./gpt-oss-20b-bf16"
out_dir = "test-20b-bf16"

model_path = "./gpt-oss-120b-int4-AutoRound"
# model_path = "./gpt-oss-120b-int4-rtn-AutoRound"
ori_model_path = "./gpt-oss-120b-bf16"
out_dir = "test-120b-bf16"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

state_dict = model.state_dict()
# print(state_dict.keys())
new_state_dict = {}

for key in state_dict:
    if "up_projs" in key:
        continue
    if "down_projs" in key:
        continue
    if "router" in key:
        new_key = key.replace("router.router", "router")
        print(new_key)
        new_state_dict[new_key] = state_dict[key]
        continue

    new_state_dict[key] = state_dict[key]


config = AutoConfig.from_pretrained(model_path)

print(model)


for name, mod in model.named_modules():
    if "gate_up_projs" in name:
        if isinstance(mod, torch.nn.ModuleList):
            print(f"old name: {name}")
            new_name = ".".join(name.split(".")[:-1])
            new_w = new_name + ".gate_up_proj"
            new_b = new_name + ".gate_up_proj_bias"
            print(f"new name: {new_name}")

            new_weight = torch.empty(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size)
            new_bias = torch.empty(config.num_local_experts, 2 * config.intermediate_size)
            for index, sub_mod in enumerate(mod):
                # print(index)
                # print(sub_mod)
                de_weight, bias = sub_mod.unpack()
                # print(de_weight.shape)
                new_weight[index, :, :].copy_(de_weight)
                # print(bias.shape)
                new_bias[index].copy_(bias)

            new_state_dict[new_w] = new_weight
            new_state_dict[new_b] = new_bias
    if "down_projs" in name:
        if isinstance(mod, torch.nn.ModuleList):
            print(f"old name: {name}")
            new_name = ".".join(name.split(".")[:-1])
            new_w = new_name + ".down_proj"
            new_b = new_name + ".down_proj_bias"
            print(f"new name: {new_name}")

            new_weight = torch.empty(config.num_local_experts, config.intermediate_size, config.hidden_size)
            new_bias = torch.empty(config.num_local_experts, config.hidden_size)
            for index, sub_mod in enumerate(mod):
                # print(index)
                # print(sub_mod)
                de_weight, bias = sub_mod.unpack()
                # print(de_weight.shape)
                new_weight[index, :, :].copy_(de_weight)
                # print(bias.shape)
                new_bias[index].copy_(bias)

            new_state_dict[new_w] = new_weight
            new_state_dict[new_b] = new_bias


from modeling_gpt_oss_official import GptOssForCausalLM


ori_config = AutoConfig.from_pretrained(ori_model_path)
tokenizer = AutoTokenizer.from_pretrained(ori_model_path)

#with init_empty_weights():
#    new_model = GptOssForCausalLM._from_config(ori_config)
new_model = GptOssForCausalLM.from_pretrained(ori_model_path, config=ori_config, torch_dtype=torch.bfloat16)
print(new_model)
ori_mean = {}
for name, param in new_model.named_parameters():
    ori_mean[name] = param.mean()
    if name == "model.layers.0.mlp.experts.down_proj":
        print(param)
    param.data.zero_()

for name, param in new_model.named_parameters():
    if name == "model.layers.0.mlp.experts.down_proj":
        print(param)

for key in new_state_dict:
    new_state_dict[key].to(torch.bfloat16)

new_model.load_state_dict(new_state_dict)

for name, param in new_model.named_parameters():
    print(f"diff: {ori_mean[name] - param.mean()}")
    if name == "model.layers.0.mlp.experts.down_proj":
        print(param)

new_model.save_pretrained(out_dir)
ori_config.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)


