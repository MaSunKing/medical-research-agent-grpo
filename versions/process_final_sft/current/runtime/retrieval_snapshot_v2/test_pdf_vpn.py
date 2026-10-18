"""No API calls: fake-IP resolution, pinned TLS, redirects and bad PDF fallback."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'backend/agent'))
from dr_agent.mcp_backend.apis import mineru_cloud_legacy_v54 as client
from dr_agent.mcp_backend.apis.medical_web_apis import is_readable_medical_web_content
from dr_agent.mcp_backend.apis.browse_preprocess import retrieval_spans

def dns(ip):return [(None,None,None,None,(ip,443))]
def answer(values):
    return Mock(status_code=200,json=lambda:{'Status':0,'Answer':[
        {'type':1 if ':' not in ip else 28,'data':ip} for ip in values]})

class PDFVPNTests(unittest.TestCase):
    def test_mineru_html_table_separate_from_prose(self):
        text='Assess GFR annually. <table><tr><td>Outcome</td><td>Events</td></tr><tr><td>DKA</td><td>56</td></tr></table> Follow up.'
        spans,excluded=retrieval_spans('Results',text)
        self.assertEqual([s['kind'] for s in spans],['body','table','body'])
        self.assertTrue(text[spans[1]['start_char']:spans[1]['end_char']].endswith('</table>'))
        self.assertEqual(excluded,[])

    def test_unclosed_html_table_is_auditable_not_evidence(self):
        text='Assess GFR annually. <table><tr><td>Outcome</td>'
        spans,excluded=retrieval_spans('Results',text)
        self.assertEqual(len(spans),1)
        self.assertNotIn('<table',text[spans[0]['start_char']:spans[0]['end_char']])
        self.assertEqual(excluded[0]['reason'],'unclosed_html_table')
    def test_fake_ip_requires_public_doh(self):
        with patch.object(client.socket,'getaddrinfo',return_value=dns('198.18.1.1')), patch.object(
            client.requests,'get',side_effect=[answer(['8.8.8.8']),answer(['2606:4700:4700::1111'])]):
            self.assertEqual(client.public_addresses('https://example.org/p.pdf'),
                             ['8.8.8.8','2606:4700:4700::1111'])

    def test_private_real_dns_rejected_even_for_known_origin(self):
        for ip in ('127.0.0.1','10.0.0.1','169.254.169.254','192.168.1.1','::1'):
            with self.subTest(ip=ip), patch.object(client.socket,'getaddrinfo',return_value=dns(ip)):
                with self.assertRaises(client.MinerUCloudError):client.public_url('https://www.nice.org.uk/p.pdf')

    def test_fake_ip_private_public_lookup_rejected(self):
        with patch.object(client.socket,'getaddrinfo',return_value=dns('198.18.1.1')), patch.object(
            client.requests,'get',side_effect=[answer(['10.0.0.1']),answer([])]):
            with self.assertRaises(client.MinerUCloudError):client.public_url('https://example.org/p.pdf')

    def test_resolver_failure_no_bypass(self):
        with patch.object(client.socket,'getaddrinfo',return_value=dns('198.18.1.1')), patch.object(
            client.requests,'get',return_value=Mock(status_code=503)):
            with self.assertRaises(client.MinerUCloudError):client.public_url('https://example.org/p.pdf')

    def test_one_transport_fallback(self):
        with patch.object(client.socket,'getaddrinfo',return_value=dns('198.18.1.1')), patch.object(
            client.requests,'get',side_effect=[Mock(status_code=503),answer(['8.8.8.8']),answer([])]) as get:
            self.assertEqual(client.public_addresses('https://example.org/p.pdf'),['8.8.8.8'])
            self.assertEqual(get.call_count,3)
            self.assertEqual(get.call_args.args[0],'https://dns.google/resolve')

    def test_private_answer_never_tries_alternate(self):
        with patch.object(client.socket,'getaddrinfo',return_value=dns('198.18.1.1')), patch.object(
            client.requests,'get',return_value=answer(['10.0.0.1'])) as get:
            with self.assertRaises(client.MinerUCloudError):client.public_url('https://example.org/p.pdf')
            self.assertEqual(get.call_count,1)

    def test_literals_and_url_credentials_rejected(self):
        for url in ('https://198.18.1.1/x','http://example.org/x','https://user:pass@example.org/x',
                    'https://example.org:8080/x','https://localhost/x'):
            with self.subTest(url=url),patch.object(client.socket,'getaddrinfo',return_value=dns('198.18.1.1')):
                with self.assertRaises(client.MinerUCloudError):client.public_url(url)

    def test_download_pins_ip_preserves_tls_name_and_no_auth(self):
        response=Mock(status=200)
        response.stream.return_value=[b'%PDF-example']
        pool=Mock()
        pool.request.return_value=response
        with patch.object(client,'public_addresses',return_value=['8.8.8.8']),patch.object(
            client.urllib3,'HTTPSConnectionPool',return_value=pool) as factory:
            instance=client.MinerUCloudClient.__new__(client.MinerUCloudClient)
            self.assertEqual(instance.download('https://example.org/p.pdf'),b'%PDF-example')
            self.assertEqual(factory.call_args.args,('8.8.8.8',443))
            self.assertEqual(factory.call_args.kwargs['server_hostname'],'example.org')
            self.assertEqual(factory.call_args.kwargs['cert_reqs'],'CERT_REQUIRED')
            self.assertEqual(pool.request.call_args.kwargs['headers'],{'Host':'example.org'})

    def test_private_redirect_is_rechecked(self):
        pool=Mock()
        pool.request.return_value=Mock(status=302,headers={'Location':'https://127.0.0.1/x'})
        with patch.object(client,'public_addresses',side_effect=[['8.8.8.8'],client.MinerUCloudError('non_public_url')]),patch.object(
            client.urllib3,'HTTPSConnectionPool',return_value=pool):
            instance=client.MinerUCloudClient.__new__(client.MinerUCloudClient)
            with self.assertRaises(client.MinerUCloudError):instance.download('https://example.org/p.pdf')
            self.assertEqual(pool.request.call_count,1)

    def test_second_validated_ip_on_connection_failure(self):
        first=Mock()
        first.request.side_effect=client.urllib3.exceptions.HTTPError('connection failed')
        second=Mock()
        second.request.return_value=Mock(status=200)
        second.request.return_value.stream.return_value=[b'%PDF-example']
        with patch.object(client,'public_addresses',return_value=['8.8.8.8','1.1.1.1']),patch.object(
            client.urllib3,'HTTPSConnectionPool',side_effect=[first,second]) as factory:
            instance=client.MinerUCloudClient.__new__(client.MinerUCloudClient)
            self.assertEqual(instance.download('https://example.org/p.pdf'),b'%PDF-example')
            self.assertEqual(factory.call_count,2)
            self.assertEqual(factory.call_args.kwargs['assert_hostname'],'example.org')

    def test_navigation_and_challenge_not_evidence(self):
        self.assertFalse(is_readable_medical_web_content(' '.join(f'Page-{n}' for n in range(1,64))))
        self.assertFalse(is_readable_medical_web_content(' '.join(f'link to page {n}' for n in range(1,30))))
        self.assertFalse(is_readable_medical_web_content('Client Challenge\nA required part of this site couldn\u2019t load.'+' Network issues.'*20))
        prose='Assess eGFR and albuminuria at least annually in patients with CKD. More frequent monitoring may be appropriate in higher-risk patients. '
        self.assertTrue(is_readable_medical_web_content(prose*4))
        self.assertTrue(is_readable_medical_web_content('Page-1 Page-2 Page-3 '+prose*4))

if __name__=='__main__':unittest.main()
