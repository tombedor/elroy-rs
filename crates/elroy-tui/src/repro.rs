
#[cfg(test)]
mod reproduction_tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    #[test]
    fn test_long_input_is_truncated_without_wrap() {
        let mut app = TuiApp::bootstrap();
        // Width 10, so "01234567890123456789" should wrap into 2 lines if wrapping is enabled.
        // Input box height will be 4 (2 lines + 2 borders).
        app.input = "01234567890123456789".to_string();
        
        let area = Rect::new(0, 0, 10, 10);
        let mut buf = Buffer::empty(area);
        app.render(area, &mut buf);
        
        // Let's see what's in the input box.
        // input_box_height(10, 10) -> wrapped_line_count("...", 8) -> 20.div_ceil(8) = 3.
        // height = 3 + 2 = 5.
        // vertical layout: 
        // [0] Min(1) -> height 4
        // [1] Length(5) -> height 5
        // [2] Length(1) -> height 1
        
        // Input box is at row 4, 5, 6, 7, 8.
        // Content should be at row 5, 6, 7.
        
        let row5 = (0..10).map(|c| buf.get(c, 5).symbol()).collect::<String>();
        let row6 = (0..10).map(|c| buf.get(c, 6).symbol()).collect::<String>();
        
        println!("Row 5: '{}'", row5);
        println!("Row 6: '{}'", row6);
        
        // If it doesn't wrap, row 5 will have "012345678" (truncated) and row 6 will be empty.
        // (Actually it will be "│01234567│" because of borders)
    }
}
