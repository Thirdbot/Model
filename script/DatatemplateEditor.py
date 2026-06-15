"""
Template and dataset alignment for compatibility, for now just only set up template
"""
import json
from pathlib import Path

from configs import load_config

class Template:
    def __init__(self,tokenizer,dataset,model_name,dataset_name,key_map,key_owner,system_message=""):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.set_add_generation_prompt = False
        self.set_tokenize = False

        self.root_path = Path(load_config("paths")['root']).resolve()
        self.template_save_path = self.root_path / load_config("paths")['dirs']['template'] / "my_template.jinja"
        self.map_config_save_path = self.root_path / load_config("paths")['dirs']['template']

        self.system_message = system_message
        self.system_role,self.user_role,self.assistant_role = key_owner.keys()

        self.key_map = key_map
        self.key_owner = key_owner
        self.modal_keys_save_path = self.map_config_save_path / self.model_name / f"{self.dataset_name}_modals.json"
        self.owner_keys_save_path = self.map_config_save_path / self.model_name / f"{self.dataset_name}_owner.json"
        self._set_key_mapping(self.key_map,self.modal_keys_save_path) # map keys to category
        self._set_key_mapping(self.key_owner,self.owner_keys_save_path) # map keys to system | user | assistant
        self.dataset = self._set_dataset_templated() # set template for dataset
        self.tokenized_dataset = self.dataset.map(self._formatting_prompts_func, batched=True)

    def solve(self):
        pass

    def _set_key_mapping(self,keys,path):
        for key,value in keys.items():
            setattr(self,key,value)
        self._save_map(keys,save_path=path)

    @staticmethod
    def _save_map(available_map,save_path=None):
        if not Path(save_path).exists():
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            Path(save_path).touch()
        with open(save_path,"w") as f:
            f.write(json.dumps(available_map))
        print("saved mapping to",save_path)

    def _template_solver(self):
        """ 1-1 mapping template to dataset"""
        pass

    @staticmethod
    def _collect(data,contents):
       return [ contents[key] for key in data if key in contents.keys()]

    def _message_to_template(self,example):
        packed_data,resolve = self._message_solver(example)
        system_owner = getattr(self,'system')
        user_owner = getattr(self,'user')
        assistant_owner = getattr(self,'assistant')

        packed_data['system_template']['content'] = self._collect(system_owner,resolve)
        packed_data['user_template']['content'] = self._collect(user_owner,resolve)
        packed_data['assistant_template']['content'] = self._collect(assistant_owner,resolve)

        extend_data = {"messages":[value for value in packed_data.values() if value is not None]}
        return extend_data


    def _message_solver(self,example,system=None,user=None,assistant=None):
        text = getattr(self,'text')
        image = getattr(self,'image')

        text_content_resolve = {f"{text_col}": {"type":"text","text":example[text_col]} for text_col in text}
        image_content_resolve = {f"{image_col}": {"type":"image"} for image_col in image}

        extends_content = text_content_resolve | image_content_resolve


        system = system or {
                "role":self.system_role,
                "content":self.system_message,
            }
        user = user or {
                "role":self.user_role,
                "content":None
            }
        assistant = assistant or {
                "role":self.assistant_role,
                "content":None,
            }
        return {
            "system_template": system,
            "user_template": user,
            "assistant_template": assistant,
        },extends_content

    def _set_dataset_templated(self):
        return self._dataset_format_to_template(self._message_to_template)

    def _dataset_format_to_template(self,message):
        return self.dataset.map(message)

    def _formatting_prompts_func(self,examples):
        convos = examples["messages"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=self.set_tokenize, add_generation_prompt=self.set_add_generation_prompt) for convo in convos]
        return {"text": texts, }


if __name__ == "__main__":
    from pandas import read_csv
    from script.HuggingfaceDownload import solve_model,solve_dataset

    model_solver, loaded_model = solve_model("geshang/Seg-R1-3B",
                                             load_in_n_bit=16,
                                             unsloth_mode=False)
    model,tokenizer = loaded_model[:2]
    dataset_path = "/home/third/Desktop/simulationv2/Dataset/multimodal_multi_image_dataset.csv"
    # dataset = read_csv(dataset_path)
    dataset_solver, dataset = solve_dataset(
        "geshang/FCoT",
    )
    dataset = dataset['train']

    key_map = {
        "image": ["image"],
        "text": ["thinking", "problem", "solution"],
    }

    key_owner = {
        "system": ["system_prompt"],
        "user": ["problem", "image"],
        "assistant": ["thinking", "solution"],
    }

    template = Template(dataset=dataset,tokenizer=tokenizer,model_name="geshang/Seg-R1-3B",dataset_name="geshang/FCoT",key_map=key_map,key_owner=key_owner)

    print(template.tokenized_dataset[0])