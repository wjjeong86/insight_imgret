import os, sys
import logging
from logging import handlers


# https://yurimkoo.github.io/python/2019/08/11/logging.html
# https://docs.python.org/ko/3/howto/logging.html
# https://jusths.tistory.com/1  <--- 이걸로 구현 함.
# https://github.com/google-research/vision_transformer/blob/master/vit_jax/logging.py < 참고



def setup_logger(log_dir,log_name):
    
    if setup_logger.log != None:
        return setup_logger.log
    
    os.makedirs(log_dir, exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    TRFH = handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir,f'{log_name}.txt'), when='midnight', interval=1, encoding='utf-8'
    )
    TRFH.setFormatter(formatter)
    TRFH.suffix = '%Y%m%d'
    TRFH.setLevel(logging.INFO)
    
    SH = logging.StreamHandler(sys.stdout)
    SH.setFormatter(formatter)
    SH.setLevel(logging.DEBUG)
    
    log = logging.getLogger('name')
    log.setLevel(logging.DEBUG)
    log.addHandler(TRFH)
    log.addHandler(SH)
    
    setup_logger.log = log
    return setup_logger.log

setup_logger.log = None

    
if __name__ == "__main__" :
    log = setup_logger("./")
    
    log.info('ddd')
    log.debug('debug')
