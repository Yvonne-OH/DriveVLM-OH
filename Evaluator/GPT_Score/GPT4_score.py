import openai
import os
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator


class GPT_SCORE:
    def __init__(self, Api_key: str = None, model: str = None, Temperature: float = 0.5, Top_p: float = 0.3, **kwargs):
        """
        Initialize the GPT_SCORE class with parameters for GPT or LLaMA scoring.

        Args:
            Api_key (str): API key for GPT scoring (if using GPT).
            model (str): Model name to use for GPT or LLaMA.
            Temperature (float): Temperature parameter for response generation.
            Top_p (float): Top-p parameter for response generation.
        """
        self.api_key = Api_key
        self.model = model
        self.temperature = Temperature
        self.top_p = Top_p

        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        if model and "GPT" in model.upper():
            if not self.api_key:
                raise ValueError("API key is required for GPT scoring.")
            self.scorer_type = "GPT"
        elif model and "LLAMA" in model.upper():
            self.scorer_type = "LLAMA"
        else:
            raise ValueError("Invalid model name. ")

    def Llama_Eval_init(self, model_name: str, finetuning_path: str = None, max_memory: dict = None, **kwargs):
        """
        Initialize the LLaMA model for evaluation.

        Args:
            model_name (str): Name or path of the LLaMA model.
            finetuning_path (str): Path to the LoRA fine-tuning adapter (if any).
            max_memory (dict): Maximum memory allocation for each device.
            kwargs: Additional parameters for model initialization.
        """
        assert model_name is not None, "Please provide a model name for evaluation."

        accelerator = Accelerator()

        # Step 1: Print model loading message
        print(f"Loading model: {model_name}")

        # Step 2: Configure quantization options, check if provided in kwargs
        bnb_config = kwargs.get("bnb_config", BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ))

        # Step 3: Set device_map and torch_dtype
        device_map = kwargs.get("device_map", "auto" if torch.cuda.is_available() else None)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        try:
            # Step 4: Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_safetensors=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map,
                max_memory=max_memory,
            ).eval()

            # Step 5: Load LoRA fine-tuning adapter (if provided)
            if finetuning_path and os.path.exists(finetuning_path):
                from peft import PeftModel

                print(f"Loading LoRA adapter from '{finetuning_path}'...")
                model = PeftModel.from_pretrained(
                    model,
                    finetuning_path,
                    is_adapter=True,
                    load_in_8bit=True,
                )
                print("LoRA adapter merged successfully")

            # Step 6: Optimize model and tokenizer using Accelerator
            model, tokenizer = accelerator.prepare(model, tokenizer)

            # Step 7: Print success message
            print("Model and tokenizer are ready for use.")

        except Exception as e:
            raise e

        self.model = model
        self.tokenizer = tokenizer

    def GPT_Score_interface(self, MCQs_Evaluation_text: str) -> str:
        """
        Evaluate the input text using the GPT model through the OpenAI API.

        Args:
            MCQs_Evaluation_text (str): Input text for evaluation.

        Returns:
            str: The GPT model's response.
        """
        openai.api_key = self.api_key
        try:
            client = OpenAI(
                api_key=self.api_key,
            )

            # Format input as a list of messages
            messages = [
                {"role": "system", "content": "Please give your rating based on the documents given."},
                {"role": "user", "content": MCQs_Evaluation_text}
            ]
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise e

    def Llama_Score_interface(self, text_input: str) -> str:
        """
        Evaluate the input text using the LLaMA model.

        Args:
            text_input (str): The text input to evaluate.

        Returns:
            str: The generated response from the LLaMA model.
        """
        try:
            assert hasattr(self, 'model') and hasattr(self,
                                                      'tokenizer'), "LLaMA model and tokenizer must be initialized."

            messages = [
                {"role": "system", "content": "Please give your rating based on the documents given."},
                {"role": "user", "content": text_input},
            ]

            # Convert messages to input format
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Set terminators
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate output
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Decode the generated output
            response = outputs[0][input_ids.shape[-1]:]  # Extract newly generated tokens
            generated_text = self.tokenizer.decode(response, skip_special_tokens=True)

            return generated_text

        except Exception as e:
            raise e

    def score(self, input_text: str) -> str:
        """
        Unified interface for scoring based on the selected model.

        Args:
            input_text (str): The text input to evaluate.

        Returns:
            str: The generated response from the selected model.
        """
        if self.scorer_type == "GPT":
            return self.GPT_Score_interface(input_text)
        elif self.scorer_type == "LLAMA":
            return self.Llama_Score_interface(input_text)
        else:
            raise ValueError("Invalid scorer type. Ensure the model is properly initialized.")


if __name__ == "__main__":
    # Suppress logging warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
    os.environ["http_proxy"] = "http://localhost:7897"
    os.environ["https_proxy"] = "http://localhost:7897"

    with open("MCQs_Eval.txt", "r") as f:
        template = f.read()

    Question = """A self-driving car encounters a complex intersection with multiple objects and conditions to consider. 
                             Based on the rules below, identify the car's primary decision in the scenario:
                            Scenario:
                            The car approaches an intersection with a stop sign on the right and a pedestrian crossing from left to right. 
                            A cyclist is coming from the opposite direction, and the traffic light ahead is green. The weather is clear, but the sun is setting, causing glare.

                            Options:
                            A. Proceed through the intersection immediately.
                            B. Stop and yield to the pedestrian and cyclist, despite the green light.
                            C. Ignore the stop sign and focus on the green light.
                            D. Slow down, assess all conditions, and proceed cautiously while yielding to any right-of-way entities.
    """
    Desc = "The question evaluates the decision-making capabilities of a self-driving car when encountering a complex intersection, focusing on traffic rules, object recognition, environmental awareness, and reasoning clarity."
    GT = "The car should slow down, assess all conditions (stop sign, pedestrian, cyclist, traffic light, and glare), and proceed cautiously while yielding to entities with the right of way."
    PRED = """
    D. Slow down, assess all conditions, and proceed cautiously while yielding to any right-of-way entities. 
    This option ensures compliance with traffic rules by yielding to the pedestrian and cyclist, who have the right of way, even with a green traffic light.
    It demonstrates proper object recognition by considering the stop sign, pedestrian, cyclist, and traffic light, 
    while also accounting for their locations and movements. Additionally, it acknowledges the environmental factor of glare from the setting sun, 
    ensuring a safe and cautious approach in line with clear and logical reasoning."""

    MCQs_Evaluation_text = template.replace("<<QUESTION>>", Question).replace("<<DESC>>", Desc).replace("<<GT>>",
                                                                                                        GT).replace(
        "<<PRED>>", PRED)

    # Initialize LLaMA model
    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-3B-Instruct'
    finetuning_path = None
    max_memory = {0: "22GB", 1: "7.6GB"}

    Api_key =  "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"
    model = "gpt-4o-mini"  # Correct model name
    Temperature = 0.5
    Top_p = 0.3

    # Define custom parameters
    custom_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Use float16 instead of bfloat16 for testing
    )
    custom_device_map = "auto"  # Automatically allocate model across available GPUs
    custom_torch_dtype = torch.float16  # Use float16 precision for speed

    # Pass the custom parameters via kwargs
    llama_evaluator = GPT_SCORE(
        model=model_name,
        Temperature=Temperature,
        Top_p=Top_p
    )

    llama_evaluator.Llama_Eval_init(
        model_name=model_name,
        finetuning_path=finetuning_path,
        max_memory=max_memory,
        bnb_config=custom_bnb_config,
        device_map=custom_device_map,
        torch_dtype=custom_torch_dtype
    )

    # Create default LLaMA evaluator
    #llama_evaluator = GPT_SCORE(model=model_name, Temperature=Temperature, Top_p=Top_p)
    #llama_evaluator.Llama_Eval_init(model_name=model_name, finetuning_path=finetuning_path, max_memory=max_memory)

    # Perform inference using LLaMA model
    response = llama_evaluator.score(MCQs_Evaluation_text)
    print("LLaMA Model Response:")
    print(response)

    print("-" * 50)

    # Create GPT evaluator
    GPT_Evaluator = GPT_SCORE(Api_key, model, Temperature, Top_p)
    GPT_score = GPT_Evaluator.GPT_Score_interface(MCQs_Evaluation_text)
    print("GPT4O Model Response:")
    print(GPT_score)