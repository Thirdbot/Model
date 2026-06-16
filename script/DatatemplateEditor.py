"""
Template and dataset alignment for compatibility, for now just only set up template
"""
import json
from pathlib import Path

from numpy.distutils.fcompiler import none

from configs import load_config

class Template:
    def __init__(self,tokenizer,dataset,model_name,dataset_name,key_map,key_owner,system_message="",set_add_generation_prompt=False,temp_for='sft',additional_images=None,additional_tokens=None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.test_size = 0.2
        self.temp_for = temp_for
        self.additional_images = additional_images or None
        self.additional_tokens = additional_tokens or None

        self.set_add_generation_prompt = False or set_add_generation_prompt
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



    def solve(self):
        first = self.dataset.train_test_split(test_size=self.test_size, shuffle=True)
        second = first["test"].train_test_split(test_size=0.5, shuffle=True)

        train_dataset = first["train"]
        eval_dataset = second["train"]
        test_dataset = second["test"]

        dataset_columns = filter(lambda x: x not in self.additional_images,self.dataset.column_names)  if self.additional_images else self.dataset.column_names # exclude mask_images from getting removed
        train_dataset  = train_dataset.map(self._message_to_template, remove_columns=dataset_columns)
        eval_dataset = eval_dataset.map(self._message_to_template, remove_columns=dataset_columns)
        test_dataset = test_dataset.map(self._message_to_template, remove_columns=dataset_columns)

        if self.temp_for == 'grpo':
            return train_dataset,eval_dataset,test_dataset

        elif self.temp_for == 'sft':
            train_dataset = train_dataset.map(self._formatting_prompts_func, batched=True,remove_columns="messages")
            eval_dataset = eval_dataset.map(self._formatting_prompts_func, batched=True,remove_columns="messages")
            test_dataset = test_dataset.map(self._formatting_prompts_func, batched=True,remove_columns="messages")
        else:
            train_dataset = train_dataset.map(self._formatting_prompts_func, batched=True, remove_columns="messages")
            eval_dataset = eval_dataset.map(self._formatting_prompts_func, batched=True, remove_columns="messages")
            test_dataset = test_dataset.map(self._formatting_prompts_func, batched=True, remove_columns="messages")

        return train_dataset,eval_dataset,test_dataset

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
        store = []
        for key in data:
            if not key or not isinstance(key, str) or not key.strip():
                continue
            if key in contents.keys():
                if isinstance(contents[key],list):
                    store.extend(contents[key]) # de-list
                else:
                    store.append(contents[key])
        return store

    def _message_to_template(self,example):
        image_key = getattr(self, 'image')
        packed_data,resolve = self._message_solver(example)
        system_owner = getattr(self,'system')
        user_owner = getattr(self,'user')
        assistant_owner = getattr(self,'assistant')

        packed_data['system_template']['content'] = self._collect(system_owner,resolve)
        packed_data['user_template']['content'] = self._collect(user_owner,resolve)
        packed_data['assistant_template']['content'] = self._collect(assistant_owner,resolve)

        images = []
        masks = []

        for img_k in image_key:
            if img_k in self.additional_images:
                mask_value = example[img_k] # right now, handle for mask
                if isinstance(mask_value, list):
                    masks.extend(mask_value)
                else:
                    masks.append(mask_value)

            value = example[img_k]

            if isinstance(value, list):
                images.extend(value)
            else:
                images.append(value)


        if self.temp_for == 'sft':
            messages = [
                value
                for value in packed_data.values()
                if value is not None and value["content"]
            ]
            if masks:
                extend_data = {"messages":messages,"images":images,"masks":masks}
                return extend_data
            extend_data = {"messages":messages,"images":images}
            return extend_data
        elif self.temp_for == 'grpo':
            prompt = []

            if packed_data["system_template"]["content"]:
                prompt.append(packed_data["system_template"])

            prompt.append(packed_data["user_template"])

            solution = example['solution']
            extend_data = {"prompt":prompt,"images":images,"target":solution}
            return extend_data
        else:
            return None

    def _valid_key(self, key, example=None):
        if key is None:
            return False
        if not isinstance(key, str):
            return False
        if not key.strip():
            return False
        if example is not None and key not in example:
            return False
        return True

    def _message_solver(self,example,system=None,user=None,assistant=None):
        text = getattr(self,'text')
        image = getattr(self,'image')

        text_content_resolve = {f"{text_col}": {"type":"text","text":f"{example[text_col]}{'' if not self.additional_tokens else self.additional_tokens[0]}\n"} for text_col in text if self._valid_key(text_col,example)}
        image_content_resolve = {f"{image_col}": [{"type":"image"} for _ in range(0,len(example[image_col]) if isinstance(example[image_col],list) else 1)] for image_col in image if example[image_col] is not None}

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


    def _formatting_prompts_func(self,examples):
        convos = examples["messages"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=self.set_tokenize, add_generation_prompt=self.set_add_generation_prompt) for convo in convos]
        return {"text": texts,}


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
        # "SakanaAI/JA-Multi-Image-VQA" #,
        "geshang/FCoT"
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
    train_dataset,eval_dataset,test_dataset = template.solve()
    print(f"{train_dataset[0]}\n\n{eval_dataset[0]}\n\n{test_dataset[0]}")