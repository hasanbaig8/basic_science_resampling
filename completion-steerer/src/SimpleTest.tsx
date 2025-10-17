export default function SimpleTest() {
  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Simple Test Page</h1>
      <p>If you can see this, the server is working!</p>
      <button onClick={() => alert('Click works!')}>
        Test Button
      </button>
    </div>
  );
}
