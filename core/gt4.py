import json
import hashlib
import binascii
import uuid
import re
import os
import time
import random
import logging
from typing import Dict, Any, Optional, List, Tuple
import requests
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES
from Crypto.Cipher import PKCS1_v1_5
from Crypto.Util.Padding import pad

# 从统一配置模块导入GeetestConfig
from config.settings import GeetestConfig

class GeetestV4:
    """极验V4验证码处理类"""

    def __init__(self, captcha_id: str, geetest_config: GeetestConfig, session: Optional[requests.Session] = None, cookies: Optional[Dict] = None, headers: Optional[Dict] = None):
        """
        初始化极验V4实例
        
        Args:
            captcha_id: 验证码ID
            geetest_config: GeetestConfig 配置对象
            session: requests.Session对象，可选
            cookies: 请求cookies，可选
            headers: 请求头，可选
        """
        self.logger = logging.getLogger(__name__)
        self.captcha_id = captcha_id
        self.geetest_config = geetest_config # 从settings获取的配置
        self.challenge = str(uuid.uuid4())
        self.session = session or requests.Session()
        
        # 设置默认cookie
        if not cookies:
            cookies = {
                'captcha_v4_user': str(uuid.uuid4()).replace('-', '')
            }
        self.cookies = cookies
        self.headers = self._get_default_headers()
        if headers:
            self.headers.update(headers)
        
        # 内部配置，基于GeetestConfig的default_config
        self.internal_config = self.geetest_config.default_config.copy()
        self.internal_config["captcha_id"] = captcha_id
        self.internal_config["lang"] = self.geetest_config.lang # 确保lang同步
        
        self.symmetric_key = self._generate_symmetric_key() # 修正为内部方法
        
        # 更新session的cookies
        if cookies:
            self.session.cookies.update(cookies)
        
        self.logger.info(f"GeetestV4 initialized for captcha_id: {self.captcha_id}")

    def _get_default_headers(self):
        """获取默认请求头"""
        # 可以从GeetestConfig中获取部分动态头部
        return {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": "https://gt4.geetest.com/",
            "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "script",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        }
    
    def _parse_callback_response(self, response_text: str) -> Dict:
        """解析回调函数格式的响应"""
        match = re.search(r'\((.*)\)$', response_text)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON解析失败，尝试修复: {e}")
                json_str = re.sub(r',\s*}', '}', json_str) # 移除末尾逗号
                json_str = re.sub(r',\s*]', ']', json_str) # 移除末尾逗号
                return json.loads(json_str)
        else:
            return json.loads(response_text)
    
    def load(self, captcha_id: Optional[str] = None, callback: Optional[str] = None, risk_type: str = "word", parse_response: bool = True, **kwargs) -> Dict[str, Any]:
        """加载验证码"""
        if callback is None:
            callback = f"geetest_{int(time.time() * 1000)}"
            
        url = "https://gcaptcha4.geetest.com/load"
        
        params = {
            "callback": callback,
            "captcha_id": captcha_id or self.captcha_id,
            "challenge": self.challenge,
            "client_type": self.geetest_config.client_type,
            "risk_type": risk_type,
            "lang": self.geetest_config.lang,
            "pt": self.geetest_config.default_config.get("pt", 1),
            **kwargs
        }

        response = self.session.get(url, params=params, headers=self.headers, timeout=self.geetest_config.request_timeout)
        self.logger.debug(f"Load URL: {response.url}, Status: {response.status_code}")
        
        if parse_response:
            return self._parse_callback_response(response.text)
        else:
            return response
    
    def verify(self, w: str, load_data: Dict, callback: Optional[str] = None) -> Dict[str, Any]:
        """提交验证"""
        if callback is None:
            callback = f"geetest_{int(time.time() * 1000)}"
            
        url = "https://gcaptcha4.geetest.com/verify"
        data = load_data.get("data", {})
        
        params = {
            "callback": callback,
            "captcha_id": self.captcha_id,
            "client_type": self.geetest_config.client_type,
            "lot_number": data.get("lot_number"),
            "risk_type": self.geetest_config.risk_type,
            "payload": data.get("payload"),
            "process_token": data.get("process_token"),
            "payload_protocol": self.geetest_config.default_config.get("payload_protocol", 1),
            "pt": self.geetest_config.default_config.get("pt", 1),
            "w": w
        }
        
        response = self.session.get(url, params=params, headers=self.headers, timeout=self.geetest_config.request_timeout)
        self.logger.debug(f"Verify URL: {response.url}, Status: {response.status_code}")
        self.logger.debug(f"Verify response: {response.text}")
        
        return self._parse_callback_response(response.text)
    
    def parse_load_response(self, response_data: Dict) -> Dict:
        if response_data.get("status") != "success":
            raise ValueError(f"加载失败: {response_data}")
        
        data = response_data.get("data", {})
        
        return {
            "lot_number": data.get("lot_number"),
            "pow_detail": data.get("pow_detail", {}),
            "captcha_type": data.get("captcha_type"),
            "imgs": data.get("imgs"),
            "ques": data.get("ques", []),
            "payload": data.get("payload"),
            "process_token": data.get("process_token"),
            "payload_protocol": data.get("payload_protocol", self.geetest_config.default_config.get("payload_protocol", 1)),
            "static_path": data.get("static_path"),
            "gct_path": data.get("gct_path"),
            "custom_theme": data.get("custom_theme", {})
        }
    
    def extract_image_urls(self, response_data: Dict) -> Dict[str, Optional[str]]:
        data = response_data.get("data", {})
        base_url = "https://static.geetest.com/" # 极验CDN
        main_img = f"{base_url}{data.get('imgs', '')}" if data.get('imgs') else None
        ques_imgs = [f"{base_url}{ques}" for ques in data.get("ques", [])]
        
        return {"main_img": main_img, "ques_imgs": ques_imgs}

    def _generate_symmetric_key(self) -> str:
        """生成对称密钥（16个字符）"""
        return ''.join(random.choice('0123456789abcdef') for _ in range(16))

    @staticmethod
    def generate_random4_hex():
        """生成4位随机16进制数 (辅助_generate_symmetric_key)"""
        return hex(int((1 + random.random()) * 65536) & 0xFFFF)[2:].zfill(4)

    @staticmethod
    def _parse_string_to_word_array(s: str) -> Dict:
        words = []
        for i in range(0, len(s), 4):
            w = 0
            for j in range(4):
                if i + j < len(s):
                    w |= (ord(s[i + j]) & 0xff) << (24 - j * 8)
            words.append(w)
        return {"words": words, "sigBytes": len(s)}

    @staticmethod
    def _array_to_hex(arr: List[int]) -> str:
        return ''.join(f"{b:02x}" for b in arr)

    def _rsa_encrypt_js_style(self, message: str) -> str:
        # 使用GeetestConfig中的RSA公钥
        rsa_n = int(self.geetest_config.rsa_public_key["n"], 16)
        rsa_e = int(self.geetest_config.rsa_public_key["e"], 16)
        key = RSA.construct((rsa_n, rsa_e))
        cipher = PKCS1_v1_5.new(key)
        encrypted_bytes = cipher.encrypt(message.encode('utf-8'))
        return binascii.hexlify(encrypted_bytes).decode('utf-8')

    @staticmethod
    def _aes_128_cbc_encrypt(plaintext: str, key_str: str, iv_str: str) -> List[int]:
        key_bytes = key_str.encode('utf-8')
        iv_bytes = iv_str.encode('utf-8')
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
        ct = cipher.encrypt(pad(plaintext.encode(), 16))
        return list(ct)

    def generate_w_data(self, load_response_data: Dict, passtime: int, userresponse: List[List[int]], device_id: str = "") -> Dict[str, str]:
        parsed_data = self.parse_load_response(load_response_data)
        
        # 构造加密数据
        data_to_encrypt = {
            "passtime": passtime,
            "userresponse": userresponse,
            "device_id": device_id,
            "lot_number": parsed_data["lot_number"],
            "pow_msg": "", # POW信息将在generate_w_data中填充
            "pow_sign": "",
            **self.internal_config, # 包含pt, payload_protocol, ep, biht, gee_guard, LldF, em, lang, geetest
            "lang": self.geetest_config.lang, # 确保使用最新的lang
        }

        # 如果有POW细节，生成POW数据
        if parsed_data.get("pow_detail"):
            pow_detail = parsed_data["pow_detail"]
            pow_msg, pow_sign = self._generate_pow_sign(
                self._generate_pow_msg(parsed_data["lot_number"], pow_detail),
                pow_detail["bits"]
            )
            data_to_encrypt["pow_msg"] = pow_msg
            data_to_encrypt["pow_sign"] = pow_sign
        
        json_str = json.dumps(data_to_encrypt, separators=(",", ":"))
        
        # AES加密数据
        aes_encrypted = self._aes_128_cbc_encrypt(json_str, self.symmetric_key, self.geetest_config.iv_str_hex) # 使用正确的IV
        rsa_encrypted = self._rsa_encrypt_js_style(self.symmetric_key)
        
        return {"w": self._array_to_hex(aes_encrypted) + rsa_encrypted}

    def _generate_pow_msg(self, lot_number: str, pow_detail: Dict) -> str:
        return (
            f"{pow_detail.get('version','1')}|"
            f"{pow_detail.get('bits',8)}|"
            f"{pow_detail.get('hashfunc','sha256')}|"
            f"{pow_detail['datetime']}|"
            f"{self.captcha_id}|"
            f"{lot_number}||"
        )

    def _generate_pow_sign(self, pow_msg: str, bits: int) -> Tuple[str, str]:
        base_bytes = pow_msg.encode('utf-8')
        leading_bytes = bits // 8
        remaining_bits = bits % 8
        mask = (0xFF << (8 - remaining_bits)) & 0xFF
        
        seed = int.from_bytes(os.urandom(8), 'big')
        
        i = 0
        while True:
            nonce_str = f"{(seed + i) & 0xFFFFFFFFFFFFFFFF:016x}"
            nonce_bytes = nonce_str.encode('utf-8')
            
            res_bytes = hashlib.sha256(base_bytes + nonce_bytes).digest()
            
            is_match = True
            for j in range(leading_bytes):
                if res_bytes[j] != 0:
                    is_match = False
                    break
            
            if is_match:
                if remaining_bits == 0 or (res_bytes[leading_bytes] & mask) == 0:
                    return pow_msg + nonce_str, res_bytes.hex()
            
            i += 1
            if i > 1000000:
                seed = int.from_bytes(os.urandom(8), 'big')
                i = 0

        raise RuntimeError("POW生成失败")