import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    width: 400
    height: 100
    color: "lightgray"

    // RangeSlider
    RangeSlider {
        id: rangeSlider
        anchors {
            left: parent.left  // Anchor to the left edge
            right: parent.right  // Anchor to the right edge
            top: parent.top  // Anchor to the top edge
            margins: 10  // Add some margin
        }
        height: 60

        from: 0
        to: 1


        // Update backend properties when the slider changes
        first.onValueChanged: backend.updateFirstValue(first.value)
        second.onValueChanged: backend.updateSecondValue(second.value)
    }

    // Display the current range
    Text {
        anchors.top: rangeSlider.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        text: `Range: ${rangeSlider.first.value.toFixed(1)} - ${rangeSlider.second.value.toFixed(1)}`
    }
}