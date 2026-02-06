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
from curl_cffi.requests import Session, Response
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.Util.Padding import pad

from config.settings import GeetestConfig

class GeetestV4:
    """极验V4验证码处理类 (使用 curl_cffi)"""

    def __init__(self, captcha_id: str, geetest_config: GeetestConfig, session: Optional[Session] = None, cookies: Optional[Dict] = None, headers: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.captcha_id = captcha_id
        self.geetest_config = geetest_config
        self.challenge = str(uuid.uuid4())
        
        if session:
            self.session = session
        else:
            self.session = Session(
                impersonate=self.geetest_config.impersonate_browser,
                timeout=self.geetest_config.request_timeout
            )
        
        if not cookies:
            cookies = {'captcha_v4_user': str(uuid.uuid4()).replace('-', '')}
        self.cookies = cookies
        self.session.cookies.update(self.cookies)

        self.headers = self._get_default_headers()
        if headers:
            self.headers.update(headers)
        self.session.headers.update(self.headers)
        
        self.internal_config = self.geetest_config.default_config.copy()
        self.internal_config["captcha_id"] = captcha_id
        self.internal_config["lang"] = self.geetest_config.lang
        
        self.symmetric_key = self._generate_symmetric_key()
        
        #self.logger.info(f"GeetestV4 initialized for captcha_id: {self.captcha_id} with curl_cffi impersonating {self.geetest_config.impersonate_browser}")

    def _get_default_headers(self) -> dict:
        """获取默认请求头"""
        return {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": "https://gt4.geetest.com/",
            "sec-ch-ua": '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "script",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "same-site",
            "user-agent": self.geetest_config.user_agent # Use configurable User-Agent
        }
    
    def _parse_callback_response(self, response: Response) -> Dict:
        """解析回调函数格式的响应，并在失败时提供详细的错误日志。"""
        response_text = response.text
        try:
            match = re.search(r'^\w+\((.*)\)$', response_text)
            json_str = match.group(1) if match else response_text
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"JSON decode failed. Status: {response.status_code}, "
                f"URL: {response.url}, "
                f"Response text: '{response_text[:500]}...'"
            )
            raise e
    
    def load(self, captcha_id: Optional[str] = None, callback: Optional[str] = None, risk_type: str = "word", **kwargs) -> Dict[str, Any]:
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

        response = self.session.get(url, params=params)
        self.logger.debug(f"Load URL: {response.url}, Status: {response.status_code}")
        
        return self._parse_callback_response(response)
    
    def verify(self, w: str, load_data: Dict, callback: Optional[str] = None) -> Dict[str, Any]:
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
        
        response = self.session.get(url, params=params)
        self.logger.debug(f"Verify URL: {response.url}, Status: {response.status_code}")
        
        return self._parse_callback_response(response)
    
    def parse_load_response(self, response_data: Dict) -> Dict:
        if response_data.get("status") != "success":
            raise ValueError(f"加载失败: {response_data}")
        
        return response_data.get("data", {})
    
    def extract_image_urls(self, response_data: Dict) -> Dict[str, Optional[str]]:
        data = response_data.get("data", {})
        base_url = "https://static.geetest.com/"
        main_img = f"{base_url}{data.get('imgs', '')}" if data.get('imgs') else None
        ques_imgs = [f"{base_url}{ques}" for ques in data.get("ques", [])]
        
        return {"main_img": main_img, "ques_imgs": ques_imgs}

    def _generate_symmetric_key(self) -> str:
        return ''.join(random.choice('0123456789abcdef') for _ in range(16))

    @staticmethod
    def _generate_dynamic_strings(lot_number):
        """生成动态字符串（用于混淆）"(n[11:14])+.+(n[12:14]+n[6:8])": _ᕵᕺᖀᖉ(95) """
        n = lot_number
        s = {
            n[11:15]: {
                n[12:15]+n[6:9]: n[21:29]
            }
        }
        return s
    @staticmethod
    def _rsa_encrypt_js_style(message: str, rsa_n_hex: str, rsa_e_hex: str) -> str:
        rsa_n = int(rsa_n_hex, 16)
        rsa_e = int(rsa_e_hex, 16)
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
        
        data_to_encrypt = {
            "passtime": passtime, "userresponse": userresponse, "device_id": device_id,
            "lot_number": parsed_data["lot_number"], "pow_msg": "", "pow_sign": "", **self._generate_dynamic_strings(parsed_data["lot_number"]),
            **self.internal_config
        }

        if parsed_data.get("pow_detail"):
            pow_detail = parsed_data["pow_detail"]
            pow_msg, pow_sign = self._generate_pow_sign(
                self._generate_pow_msg(parsed_data["lot_number"], pow_detail), pow_detail["bits"]
            )
            data_to_encrypt["pow_msg"] = pow_msg
            data_to_encrypt["pow_sign"] = pow_sign
        
        json_str = json.dumps(data_to_encrypt, separators=(",", ":"))
        
        aes_encrypted = self._aes_128_cbc_encrypt(json_str, self.symmetric_key, self.geetest_config.iv_str_hex)
        rsa_encrypted = self._rsa_encrypt_js_style(self.symmetric_key, self.geetest_config.rsa_public_key["n"], self.geetest_config.rsa_public_key["e"])
        
        return {"w": binascii.hexlify(bytes(aes_encrypted)).decode() + rsa_encrypted}

    def _generate_pow_msg(self, lot_number: str, pow_detail: Dict) -> str:
        return f"{pow_detail.get('version','1')}|{pow_detail.get('bits',8)}|{pow_detail.get('hashfunc','sha256')}|{pow_detail['datetime']}|{self.captcha_id}|{lot_number}||"

    def _generate_pow_sign(self, pow_msg: str, bits: int) -> Tuple[str, str]:
        base_bytes = pow_msg.encode('utf-8')
        target_prefix = "0" * (bits // 4)
        
        i = 0
        while True:
            nonce_str = f"{random.randint(0, 0xFFFFFFFFFFFF):012x}"
            m = hashlib.sha256()
            m.update(base_bytes)
            m.update(nonce_str.encode('utf-8'))
            res_hex = m.hexdigest()
            
            if res_hex.startswith(target_prefix):
                return pow_msg + nonce_str, res_hex
            i += 1
            if i > 2000000: # Safety break
                raise RuntimeError("POW generation failed after too many attempts.")
